def init_mapping(self):
    from collections import defaultdict, deque
    import math

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
    # Step 2 – Gate layer (earliest scheduling depth) via Kahn BFS       #
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
    # Step 5 – Layer-decay-weighted interaction matrix                    #
    # w(g) = (critical_path[g]+1) × exp(-alpha × gate_layer[g])          #
    # Exponentially up-weights early gates — they must be placed well     #
    # first and their mis-placement causes cascading SWAPs.               #
    # alpha is adaptive: ~10× decay from layer 0 to max_layer.           #
    # ------------------------------------------------------------------ #
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    alpha = math.log(10.0) / (max_layer + 1)

    interaction_neighbors = defaultdict(dict)
    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        w = (critical_path[g] + 1) * math.exp(-alpha * gate_layer[g])
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0.0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0.0) + w

    weighted_degree = {
        q: sum(interaction_neighbors[q].values()) for q in logical_qubits
    }

    # ------------------------------------------------------------------ #
    # Step 6 – Hardware centrality                                        #
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
    # Step 7 – Top-K logical seed pairs and top-M hardware seed pairs     #
    # ------------------------------------------------------------------ #
    pair_weights = {}
    for q1 in logical_qubits:
        for q2, w in interaction_neighbors[q1].items():
            if q1 < q2:
                pair_weights[(q1, q2)] = w

    top_logic_pairs = sorted(pair_weights, key=lambda p: pair_weights[p], reverse=True)[:3]
    if not top_logic_pairs and len(logical_qubits) >= 2:
        top_logic_pairs = [(logical_qubits[0], logical_qubits[1])]

    hw_adj_pairs = sorted(
        [(cent[p1] + cent[p2], p1, p2)
         for p1 in physical_qubits for p2 in self.backend[p1] if p1 < p2]
    )
    top_phys_pairs = [(p1, p2) for _, p1, p2 in hw_adj_pairs[:3]]

    if not top_phys_pairs or not top_logic_pairs:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 8 – Mapping quality: total weighted distance                   #
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
    # Step 9 – Greedy BFS-frontier fill from a seeded pair               #
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
                    sum(interaction_neighbors[lq].get(pl, 0.0) for pl in lq_to_phys),
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

            future_nb = sum(1 for nb in interaction_neighbors[next_lq] if nb in unplaced_set)

            def _score(phys_c, _lq=next_lq, _fn=future_nb):
                dist_cost = 0.0
                for placed_lq, placed_phys_q in lq_to_phys.items():
                    w = interaction_neighbors[_lq].get(placed_lq, 0.0)
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
    # Step 10 – Try all seed combinations, keep best by mapping cost     #
    # ------------------------------------------------------------------ #
    best_cost = float('inf')
    best_lq_to_phys = None

    rem_phys_pool = sorted(
        physical_qubits, key=lambda p: len(self.backend[p]), reverse=True
    )

    for sq1, sq2 in top_logic_pairs:
        for sp1, sp2 in top_phys_pairs:
            for a, b in [(sq1, sq2), (sq2, sq1)]:
                candidate = greedy_fill(a, b, sp1, sp2)
                # Assign leftover logical qubits
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
    # Step 11 – Local swap hill-climbing                                  #
    # For each pair of logical qubits, compute the delta cost of          #
    # swapping their physical assignments in O(n) and accept if < 0.      #
    # Repeat for up to MAX_ROUNDS passes.                                 #
    # ------------------------------------------------------------------ #
    lq_list = logical_qubits[:]
    n_lq = len(lq_list)
    MAX_ROUNDS = 4

    for _ in range(MAX_ROUNDS):
        improved = False
        for i in range(n_lq):
            for j in range(i + 1, n_lq):
                lq1, lq2 = lq_list[i], lq_list[j]
                p1, p2 = lq_to_phys.get(lq1), lq_to_phys.get(lq2)
                if p1 is None or p2 is None:
                    continue

                # Delta from swapping lq1↔lq2 physical assignments.
                # The lq1–lq2 term is invariant (same two physicals, different labels).
                # Only cross-terms with all other placed qubits change.
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
                    # before: w1*d1o + w2*d2o  |  after: w1*d2o + w2*d1o
                    delta += (w1 * d2o + w2 * d1o) - (w1 * d1o + w2 * d2o)

                if delta < -1e-9:
                    lq_to_phys[lq1] = p2
                    lq_to_phys[lq2] = p1
                    improved = True
        if not improved:
            break

    # ------------------------------------------------------------------ #
    # Step 12 – Build strict 1-to-1 bijection via in-place swap chain    #
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