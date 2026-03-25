def init_mapping(self):
    from collections import defaultdict, deque
    import math
    import random

    all_gates = sorted(self.access.keys())
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]

    # ------------------------------------------------------------------ #
    # Step 1 – Build DAG (successors + predecessors)                      #
    # ------------------------------------------------------------------ #
    last_gate_on_qubit = {}
    successors_dag = defaultdict(set)
    predecessors_dag = defaultdict(set)

    for g in all_gates:
        for q in self.access[g]:
            if q in last_gate_on_qubit:
                pred = last_gate_on_qubit[q]
                successors_dag[pred].add(g)
                predecessors_dag[g].add(pred)
            last_gate_on_qubit[q] = g

    for g in all_gates:
        successors_dag.setdefault(g, set())
        predecessors_dag.setdefault(g, set())

    # ------------------------------------------------------------------ #
    # Step 2 – Gate layer via Kahn BFS                                    #
    # ------------------------------------------------------------------ #
    in_degree = {g: len(predecessors_dag[g]) for g in all_gates}
    gate_layer = {g: 0 for g in all_gates}
    temp_in = dict(in_degree)
    queue = deque(g for g in all_gates if in_degree[g] == 0)
    topo_order = []

    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in successors_dag[g]:
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            temp_in[s] -= 1
            if temp_in[s] == 0:
                queue.append(s)

    # ------------------------------------------------------------------ #
    # Step 3 – Critical path (remaining depth from each gate)             #
    # ------------------------------------------------------------------ #
    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors_dag[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # ------------------------------------------------------------------ #
    # Step 4 – Collect logical / physical qubits                          #
    # ------------------------------------------------------------------ #
    logical_qubit_set = set()
    for qubits in self.access.values():
        logical_qubit_set.update(qubits)

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits or not two_qubit_gates:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 5 – Multi-window weighted interaction matrix                   #
    # Blends a fast (alpha) early-heavy decay and a slow (beta)           #
    # circuit-wide decay so both front-loaded and mid-circuit structure   #
    # are captured in the placement cost.                                  #
    # ------------------------------------------------------------------ #
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    alpha = math.log(10.0) / (max_layer + 1)   # steep: early gates dominate
    beta  = math.log(3.0)  / (max_layer + 1)   # gentle: mid-circuit weight

    interaction_neighbors = defaultdict(dict)
    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        cp = critical_path[g] + 1
        gl = gate_layer[g]
        w = cp * (0.7 * math.exp(-alpha * gl) + 0.3 * math.exp(-beta * gl))
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0.0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0.0) + w

    # ------------------------------------------------------------------ #
    # Step 6 – Second-order interactions (A↔C via shared neighbour B)     #
    # Captures transitive pressure: if A–B–C is a sequence, placing A    #
    # near C reduces cascading SWAPs even if A–C never share a gate.      #
    # Limited to top-5 neighbours per qubit to keep runtime bounded.      #
    # ------------------------------------------------------------------ #
    second_order = defaultdict(dict)
    for mid in logical_qubits:
        top_nbs = sorted(
            interaction_neighbors[mid].keys(),
            key=lambda q: interaction_neighbors[mid][q],
            reverse=True
        )[:5]
        for i in range(len(top_nbs)):
            for j in range(i + 1, len(top_nbs)):
                qa, qb = top_nbs[i], top_nbs[j]
                w = 0.12 * math.sqrt(
                    interaction_neighbors[mid][qa] * interaction_neighbors[mid][qb]
                )
                second_order[qa][qb] = second_order[qa].get(qb, 0.0) + w
                second_order[qb][qa] = second_order[qb].get(qa, 0.0) + w

    # Effective interaction = direct + second-order
    effective = defaultdict(dict)
    for q in logical_qubits:
        for nb, w in interaction_neighbors[q].items():
            effective[q][nb] = w
        for nb, w in second_order[q].items():
            effective[q][nb] = effective[q].get(nb, 0.0) + w

    weighted_degree = {q: sum(effective[q].values()) for q in logical_qubits}

    # ------------------------------------------------------------------ #
    # Step 7 – Hardware centrality (lower = more central)                 #
    # ------------------------------------------------------------------ #
    def _centrality(p):
        finite = [
            self.distance_matrix[p][o]
            for o in physical_qubits
            if o != p and self.distance_matrix[p][o] != float('inf')
        ]
        return sum(finite) / len(finite) if finite else float('inf')

    cent = {p: _centrality(p) for p in physical_qubits}

    # ------------------------------------------------------------------ #
    # Step 8 – Mapping cost (uses effective interactions)                 #
    # ------------------------------------------------------------------ #
    def mapping_cost(lq_to_phys_map):
        total = 0.0
        lqs = list(lq_to_phys_map.keys())
        for i in range(len(lqs)):
            for j in range(i + 1, len(lqs)):
                w = effective[lqs[i]].get(lqs[j], 0.0)
                if w > 0.0:
                    d = self.distance_matrix[lq_to_phys_map[lqs[i]]][lq_to_phys_map[lqs[j]]]
                    total += w * (d if d != float('inf') else 1e9)
        return total

    # ------------------------------------------------------------------ #
    # Step 9 – Shared delta helper (used by both SA and hill-climbing)    #
    # ------------------------------------------------------------------ #
    def swap_delta(lq_to_phys, lq1, lq2):
        """Cost change from swapping phys assignments of lq1 and lq2."""
        p1, p2 = lq_to_phys[lq1], lq_to_phys[lq2]
        delta = 0.0
        for other_lq, other_phys in lq_to_phys.items():
            if other_lq == lq1 or other_lq == lq2:
                continue
            w1 = effective[lq1].get(other_lq, 0.0)
            w2 = effective[lq2].get(other_lq, 0.0)
            if w1 == 0.0 and w2 == 0.0:
                continue
            d1o = self.distance_matrix[p1][other_phys]
            d2o = self.distance_matrix[p2][other_phys]
            d1o = d1o if d1o != float('inf') else 1e9
            d2o = d2o if d2o != float('inf') else 1e9
            delta += (w1 * d2o + w2 * d1o) - (w1 * d1o + w2 * d2o)
        return delta

    # ------------------------------------------------------------------ #
    # Step 10 – Greedy BFS-frontier fill from a seeded pair               #
    # ------------------------------------------------------------------ #
    def greedy_fill(seed_lq1, seed_lq2, phys1, phys2):
        lq_to_phys = {seed_lq1: phys1, seed_lq2: phys2}
        placed_phys = {phys1, phys2}
        unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]

        while unplaced:
            unplaced_set = set(unplaced)
            next_lq = max(
                unplaced,
                key=lambda lq: (
                    sum(effective[lq].get(pl, 0.0) for pl in lq_to_phys),
                    weighted_degree.get(lq, 0.0),
                )
            )

            candidates = list({
                nb for phys in placed_phys
                for nb in self.backend[phys]
                if nb not in placed_phys
            })
            if not candidates:
                candidates = [p for p in physical_qubits if p not in placed_phys]
            if not candidates:
                break

            future_nb = sum(1 for nb in effective[next_lq] if nb in unplaced_set)

            def _score(phys_c, _lq=next_lq, _fn=future_nb):
                dist_cost = 0.0
                for placed_lq, placed_phys_q in lq_to_phys.items():
                    w = effective[_lq].get(placed_lq, 0.0)
                    if w > 0.0:
                        d = self.distance_matrix[phys_c][placed_phys_q]
                        dist_cost += w * (d if d != float('inf') else 1e9)
                empty_nb = sum(1 for nb in self.backend[phys_c] if nb not in placed_phys)
                return (dist_cost, max(0, _fn - empty_nb), -empty_nb)

            best_phys = min(candidates, key=_score)
            lq_to_phys[next_lq] = best_phys
            placed_phys.add(best_phys)
            unplaced.remove(next_lq)

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 11 – Top-K seed pairs (5 logic × 5 hardware)                  #
    # ------------------------------------------------------------------ #
    pair_weights = {}
    for q1 in logical_qubits:
        for q2, w in interaction_neighbors[q1].items():
            if q1 < q2:
                pair_weights[(q1, q2)] = w

    top_logic_pairs = sorted(pair_weights, key=lambda p: pair_weights[p], reverse=True)[:5]
    if not top_logic_pairs and len(logical_qubits) >= 2:
        top_logic_pairs = [(logical_qubits[0], logical_qubits[1])]

    hw_adj_pairs = sorted(
        [(cent[p1] + cent[p2], p1, p2)
         for p1 in physical_qubits for p2 in self.backend[p1] if p1 < p2]
    )
    top_phys_pairs = [(p1, p2) for _, p1, p2 in hw_adj_pairs[:5]]

    if not top_phys_pairs or not top_logic_pairs:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 12 – Generate all greedy candidates; keep top-3 for SA        #
    # ------------------------------------------------------------------ #
    rem_phys_pool = sorted(
        physical_qubits, key=lambda p: len(self.backend[p]), reverse=True
    )

    candidate_heap = []
    for sq1, sq2 in top_logic_pairs:
        for sp1, sp2 in top_phys_pairs:
            for a, b in [(sq1, sq2), (sq2, sq1)]:
                cand = greedy_fill(a, b, sp1, sp2)
                placed_set = set(cand.values())
                rem_p = [p for p in rem_phys_pool if p not in placed_set]
                rem_l = [lq for lq in logical_qubits if lq not in cand]
                for lq, phys in zip(rem_l, rem_p):
                    cand[lq] = phys
                candidate_heap.append((mapping_cost(cand), dict(cand)))

    candidate_heap.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------ #
    # Step 13 – Simulated Annealing on top-3 greedy candidates           #
    # SA escapes local optima the hill-climber cannot reach.              #
    # Temperature schedule: T_init ∝ initial_cost, geometric cooling.    #
    # ------------------------------------------------------------------ #
    def simulated_annealing(lq_to_phys, init_cost, n_iter, rng_seed):
        rng = random.Random(rng_seed)
        current = dict(lq_to_phys)
        current_cost = init_cost
        best = dict(current)
        best_cost = current_cost
        lqs = list(current.keys())
        n = len(lqs)
        if n < 2:
            return best, best_cost

        T = max(init_cost * 0.08, 1.0)
        T_final = max(init_cost * 5e-5, 1e-4)
        cooling = (T_final / T) ** (1.0 / n_iter)

        for _ in range(n_iter):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n - 1)
            if i2 >= i1:
                i2 += 1
            lq1, lq2 = lqs[i1], lqs[i2]

            delta = swap_delta(current, lq1, lq2)
            if delta < 0.0 or rng.random() < math.exp(-delta / T):
                p1, p2 = current[lq1], current[lq2]
                current[lq1] = p2
                current[lq2] = p1
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best = dict(current)
            T *= cooling

        return best, best_cost

    # Scale SA iterations with circuit complexity
    n_lq = len(logical_qubits)
    sa_iters = max(2000, min(6000, n_lq * 200))

    best_cost = float('inf')
    best_lq_to_phys = candidate_heap[0][1] if candidate_heap else None

    for idx, (init_cost, cand) in enumerate(candidate_heap[:3]):
        refined, refined_cost = simulated_annealing(
            cand, init_cost, n_iter=sa_iters, rng_seed=idx * 31 + 7
        )
        if refined_cost < best_cost:
            best_cost = refined_cost
            best_lq_to_phys = dict(refined)

    lq_to_phys = best_lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 14 – Local 2-opt hill-climbing (final polish after SA)        #
    # More rounds than before since SA may leave small local improvements.#
    # ------------------------------------------------------------------ #
    lq_list = logical_qubits[:]
    MAX_ROUNDS = 10

    for _ in range(MAX_ROUNDS):
        improved = False
        for i in range(n_lq):
            for j in range(i + 1, n_lq):
                lq1, lq2 = lq_list[i], lq_list[j]
                if lq_to_phys.get(lq1) is None or lq_to_phys.get(lq2) is None:
                    continue
                delta = swap_delta(lq_to_phys, lq1, lq2)
                if delta < -1e-9:
                    p1, p2 = lq_to_phys[lq1], lq_to_phys[lq2]
                    lq_to_phys[lq1] = p2
                    lq_to_phys[lq2] = p1
                    improved = True
        if not improved:
            break

    # ------------------------------------------------------------------ #
    # Step 15 – Build strict 1-to-1 bijection via in-place swap chain    #
    # ------------------------------------------------------------------ #
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)