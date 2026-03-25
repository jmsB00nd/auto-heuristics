def init_mapping(self):
    from collections import defaultdict, deque
    import math
    import random

    random.seed(42)

    all_gates = sorted(self.access.keys())
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]

    # ------------------------------------------------------------------ #
    # Step 1 – Build DAG                                                  #
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
    # Step 2 – Topological ordering + gate layers via Kahn BFS           #
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
    # Step 3 – Critical path (remaining depth from each gate)            #
    # ------------------------------------------------------------------ #
    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors_dag[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # ------------------------------------------------------------------ #
    # Step 4 – Collect logical / physical qubits                         #
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
    # Step 5 – Layer-decay-weighted interaction matrix                   #
    # w(g) = (critical_path[g]+1) * exp(-alpha * gate_layer[g])         #
    # More aggressive alpha: ~15x decay over circuit depth               #
    # ------------------------------------------------------------------ #
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    alpha = math.log(15.0) / (max_layer + 1)

    interaction_neighbors = defaultdict(dict)
    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        w = (critical_path[g] + 1) * math.exp(-alpha * gate_layer[g])
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0.0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0.0) + w

    weighted_degree = {
        q: sum(interaction_neighbors[q].values()) for q in logical_qubits
    }

    # Precompute list of interacting pairs for SA (non-zero weight)
    interacting_pairs = [
        (q1, q2)
        for q1 in logical_qubits
        for q2 in interaction_neighbors[q1]
        if q1 < q2
    ]

    # ------------------------------------------------------------------ #
    # Step 6 – Hardware centrality + degree                              #
    # ------------------------------------------------------------------ #
    def _centrality(p):
        finite = [
            self.distance_matrix[p][o]
            for o in physical_qubits
            if o != p and self.distance_matrix[p][o] != float('inf')
        ]
        return sum(finite) / len(finite) if finite else float('inf')

    cent = {p: _centrality(p) for p in physical_qubits}
    hw_degree = {p: len(self.backend[p]) for p in physical_qubits}

    # ------------------------------------------------------------------ #
    # Step 7 – Mapping cost function                                     #
    # ------------------------------------------------------------------ #
    def mapping_cost(lq_to_phys_map):
        total = 0.0
        lqs = list(lq_to_phys_map.keys())
        for i in range(len(lqs)):
            for j in range(i + 1, len(lqs)):
                w = interaction_neighbors[lqs[i]].get(lqs[j], 0.0)
                if w > 0.0:
                    d = self.distance_matrix[lq_to_phys_map[lqs[i]]][lq_to_phys_map[lqs[j]]]
                    total += w * (d if d != float('inf') else 1e9)
        return total

    # ------------------------------------------------------------------ #
    # Step 8 – Seed selection: top-5 logic pairs × top-5 hw pairs       #
    # ------------------------------------------------------------------ #
    pair_weights = {}
    for q1 in logical_qubits:
        for q2, w in interaction_neighbors[q1].items():
            if q1 < q2:
                pair_weights[(q1, q2)] = w

    K = 5
    top_logic_pairs = sorted(pair_weights, key=lambda p: pair_weights[p], reverse=True)[:K]
    if not top_logic_pairs and len(logical_qubits) >= 2:
        top_logic_pairs = [(logical_qubits[0], logical_qubits[1])]

    # Hardware pairs: prefer adjacent pairs that are both central and high-degree
    hw_adj_pairs = sorted(
        [(cent[p1] + cent[p2] - 0.5 * (hw_degree[p1] + hw_degree[p2]), p1, p2)
         for p1 in physical_qubits for p2 in self.backend[p1] if p1 < p2]
    )
    M = 5
    top_phys_pairs = [(p1, p2) for _, p1, p2 in hw_adj_pairs[:M]]

    if not top_phys_pairs or not top_logic_pairs:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 9 – Greedy BFS fill with 1-hop future lookahead               #
    # ------------------------------------------------------------------ #
    def greedy_fill(seed_lq1, seed_lq2, phys1, phys2):
        lq_to_phys = {seed_lq1: phys1, seed_lq2: phys2}
        placed_phys = {phys1, phys2}
        unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]

        while unplaced:
            unplaced_set = set(unplaced)

            # Choose next logical qubit by interaction with placed qubits,
            # tie-break by overall weighted degree
            next_lq = max(
                unplaced,
                key=lambda lq: (
                    sum(interaction_neighbors[lq].get(pl, 0.0) for pl in lq_to_phys),
                    weighted_degree.get(lq, 0.0),
                )
            )

            # Candidate physical qubits: neighbors of placed first, then all free
            candidates = list({
                nb for phys in placed_phys
                for nb in self.backend[phys]
                if nb not in placed_phys
            })
            if not candidates:
                candidates = [p for p in physical_qubits if p not in placed_phys]
            if not candidates:
                break

            # Future unplaced qubits that interact with next_lq
            future_nb_lqs = [
                nb for nb in interaction_neighbors[next_lq]
                if nb in unplaced_set
            ]
            future_nb_weight = sum(
                interaction_neighbors[next_lq].get(nb, 0.0) for nb in future_nb_lqs
            )

            def _score(phys_c, _lq=next_lq, _fnl=future_nb_lqs, _fnw=future_nb_weight):
                # Primary: weighted distance to already-placed qubits
                dist_cost = 0.0
                for placed_lq, placed_phys_q in lq_to_phys.items():
                    w = interaction_neighbors[_lq].get(placed_lq, 0.0)
                    if w > 0.0:
                        d = self.distance_matrix[phys_c][placed_phys_q]
                        dist_cost += w * (d if d != float('inf') else 1e9)

                # Lookahead: free neighbors of phys_c can accommodate future interacting qubits
                free_nb_count = sum(
                    1 for nb in self.backend[phys_c] if nb not in placed_phys
                )
                # Penalize if we need neighbors for future qubits but don't have them
                future_penalty = max(0, len(_fnl) - free_nb_count) * _fnw * 0.5

                # Secondary: prefer leaving room for future placements
                return (dist_cost + future_penalty, -free_nb_count)

            best_phys = min(candidates, key=_score)
            lq_to_phys[next_lq] = best_phys
            placed_phys.add(best_phys)
            unplaced.remove(next_lq)

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 10 – Try all seed combinations, keep best by mapping cost     #
    # ------------------------------------------------------------------ #
    best_cost = float('inf')
    best_lq_to_phys = None

    rem_phys_pool = sorted(
        physical_qubits,
        key=lambda p: (hw_degree[p], -cent[p]),
        reverse=True
    )

    for sq1, sq2 in top_logic_pairs:
        for sp1, sp2 in top_phys_pairs:
            # Try both logic orientations; hw orientation handled by pair construction
            for a, b in [(sq1, sq2), (sq2, sq1)]:
                for pp1, pp2 in [(sp1, sp2), (sp2, sp1)]:
                    candidate = greedy_fill(a, b, pp1, pp2)
                    placed_phys_set = set(candidate.values())
                    rem_phys = [p for p in rem_phys_pool if p not in placed_phys_set]
                    rem_lqs = [lq for lq in logical_qubits if lq not in candidate]
                    for lq, phys in zip(rem_lqs, rem_phys):
                        candidate[lq] = phys

                    c = mapping_cost(candidate)
                    if c < best_cost:
                        best_cost = c
                        best_lq_to_phys = dict(candidate)

    lq_to_phys = best_lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 11 – Hill-climbing (pairwise swaps)                           #
    # ------------------------------------------------------------------ #
    lq_list = logical_qubits[:]
    n_lq = len(lq_list)

    def hill_climb(mapping, rounds=6):
        for _ in range(rounds):
            improved = False
            for i in range(n_lq):
                for j in range(i + 1, n_lq):
                    lq1, lq2 = lq_list[i], lq_list[j]
                    p1, p2 = mapping.get(lq1), mapping.get(lq2)
                    if p1 is None or p2 is None:
                        continue
                    delta = 0.0
                    for other_lq, other_phys in mapping.items():
                        if other_lq == lq1 or other_lq == lq2:
                            continue
                        w1 = interaction_neighbors[lq1].get(other_lq, 0.0)
                        w2 = interaction_neighbors[lq2].get(other_lq, 0.0)
                        if w1 == 0.0 and w2 == 0.0:
                            continue
                        d1o = self.distance_matrix[p1][other_phys]
                        d2o = self.distance_matrix[p2][other_phys]
                        d1o = d1o if d1o != float('inf') else 1e9
                        d2o = d2o if d2o != float('inf') else 1e9
                        delta += (w1 * d2o + w2 * d1o) - (w1 * d1o + w2 * d2o)
                    if delta < -1e-9:
                        mapping[lq1] = p2
                        mapping[lq2] = p1
                        improved = True
            if not improved:
                break
        return mapping

    lq_to_phys = hill_climb(lq_to_phys, rounds=6)

    # ------------------------------------------------------------------ #
    # Step 12 – Simulated Annealing (interaction-focused moves)          #
    # Operate only on pairs with non-zero interaction weight to ensure   #
    # every SA move is meaningful.                                        #
    # ------------------------------------------------------------------ #
    if interacting_pairs:
        current_cost = mapping_cost(lq_to_phys)
        best_sa_cost = current_cost
        best_sa_map = dict(lq_to_phys)

        T_init = max(current_cost * 0.05, 0.5)
        T_final = 1e-4
        n_steps = min(4000, max(1000, n_lq * n_lq * 8))
        cooling = (T_final / T_init) ** (1.0 / n_steps)
        T = T_init
        n_pairs = len(interacting_pairs)

        for _ in range(n_steps):
            # Pick a random interacting pair — moves are always relevant
            lq1, lq2 = interacting_pairs[int(random.random() * n_pairs)]
            p1, p2 = lq_to_phys.get(lq1), lq_to_phys.get(lq2)
            if p1 is None or p2 is None:
                T *= cooling
                continue

            delta = 0.0
            for other_lq, other_phys in lq_to_phys.items():
                if other_lq == lq1 or other_lq == lq2:
                    continue
                w1 = interaction_neighbors[lq1].get(other_lq, 0.0)
                w2 = interaction_neighbors[lq2].get(other_lq, 0.0)
                if w1 == 0.0 and w2 == 0.0:
                    continue
                d1o = self.distance_matrix[p1][other_phys]
                d2o = self.distance_matrix[p2][other_phys]
                d1o = d1o if d1o != float('inf') else 1e9
                d2o = d2o if d2o != float('inf') else 1e9
                delta += (w1 * d2o + w2 * d1o) - (w1 * d1o + w2 * d2o)

            if delta < 0 or (T > 1e-12 and random.random() < math.exp(-delta / T)):
                lq_to_phys[lq1] = p2
                lq_to_phys[lq2] = p1
                current_cost += delta
                if current_cost < best_sa_cost:
                    best_sa_cost = current_cost
                    best_sa_map = dict(lq_to_phys)

            T *= cooling

        lq_to_phys = best_sa_map

    # Final hill-climbing pass to clean up after SA
    lq_to_phys = hill_climb(lq_to_phys, rounds=6)

    # ------------------------------------------------------------------ #
    # Step 13 – Build strict 1-to-1 bijection via in-place swap chain   #
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