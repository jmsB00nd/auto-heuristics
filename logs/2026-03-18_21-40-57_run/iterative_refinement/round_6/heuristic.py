def init_mapping(self):
    from collections import defaultdict, deque
    import math
    import random

    rng = random.Random(42)
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
    # Step 2 – Gate layer via Kahn BFS                                   #
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
    # Step 4 – Logical / physical qubits                                 #
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
    # w(g) = (critical_path[g]+1) × exp(-alpha × gate_layer[g])         #
    # ------------------------------------------------------------------ #
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    alpha = math.log(10.0) / (max_layer + 1)

    interaction_neighbors = defaultdict(dict)
    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        w = (critical_path[g] + 1) * math.exp(-alpha * gate_layer[g])
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0.0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0.0) + w

    weighted_degree = {q: sum(interaction_neighbors[q].values()) for q in logical_qubits}

    # ------------------------------------------------------------------ #
    # Step 6 – Hardware structures                                        #
    # ------------------------------------------------------------------ #
    def _centrality(p):
        finite = [self.distance_matrix[p][o] for o in physical_qubits
                  if o != p and self.distance_matrix[p][o] != float('inf')]
        return sum(finite) / len(finite) if finite else float('inf')

    cent = {p: _centrality(p) for p in physical_qubits}
    phys_degree = {p: len(self.backend[p]) for p in physical_qubits}

    # ------------------------------------------------------------------ #
    # Step 7 – Mapping cost (weighted sum of distances)                  #
    # ------------------------------------------------------------------ #
    def mapping_cost(m):
        lqs = list(m.keys())
        total = 0.0
        for i in range(len(lqs)):
            for j in range(i + 1, len(lqs)):
                w = interaction_neighbors[lqs[i]].get(lqs[j], 0.0)
                if w > 0.0:
                    d = self.distance_matrix[m[lqs[i]]][m[lqs[j]]]
                    total += w * (d if d != float('inf') else 1e9)
        return total

    # ------------------------------------------------------------------ #
    # Step 8 – Greedy BFS-frontier fill from a seed assignment           #
    # ------------------------------------------------------------------ #
    def greedy_fill(seed_assignment):
        lq_to_phys = dict(seed_assignment)
        placed_phys = set(lq_to_phys.values())
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
    # Step 9 – Pairwise hill-climbing local search                       #
    # ------------------------------------------------------------------ #
    def hill_climb(lq_to_phys, max_rounds=6):
        lq_list = logical_qubits[:]
        n_lq = len(lq_list)
        for _ in range(max_rounds):
            improved = False
            for i in range(n_lq):
                for j in range(i + 1, n_lq):
                    lq1, lq2 = lq_list[i], lq_list[j]
                    p1 = lq_to_phys.get(lq1)
                    p2 = lq_to_phys.get(lq2)
                    if p1 is None or p2 is None:
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
                    if delta < -1e-9:
                        lq_to_phys[lq1] = p2
                        lq_to_phys[lq2] = p1
                        improved = True
            if not improved:
                break
        return lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 10 – Build diverse seeds: pairs + triplet chains              #
    # Triplet seeding exploits the heavy-hex chain structure:            #
    # map logical chain (q_a – q_mid – q_b) to hardware path            #
    # (p_a – p_mid – p_b) where p_mid is adjacent to both p_a and p_b. #
    # ------------------------------------------------------------------ #

    # --- Pair seeds (same as round 1) ---
    pair_weights = {}
    for q1 in logical_qubits:
        for q2, w in interaction_neighbors[q1].items():
            if q1 < q2:
                pair_weights[(q1, q2)] = w
    top_logic_pairs = sorted(pair_weights, key=lambda p: pair_weights[p], reverse=True)[:4]
    if not top_logic_pairs and len(logical_qubits) >= 2:
        top_logic_pairs = [(logical_qubits[0], logical_qubits[1])]

    hw_adj_pairs = sorted(
        [(cent[p1] + cent[p2], p1, p2)
         for p1 in physical_qubits for p2 in self.backend[p1] if p1 < p2]
    )
    top_phys_pairs = [(p1, p2) for _, p1, p2 in hw_adj_pairs[:4]]

    # --- Triplet seeds: find top logical chains (q_a - q_mid - q_b) ---
    triplet_seeds = []
    for q_mid in logical_qubits:
        nbs = list(interaction_neighbors[q_mid].items())
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                q_a, w_a = nbs[i]
                q_b, w_b = nbs[j]
                # Score = strength of both arms; bonus for direct q_a-q_b interaction
                w_ab = interaction_neighbors[q_a].get(q_b, 0.0)
                score = w_a + w_b + 0.3 * w_ab
                triplet_seeds.append((score, q_a, q_mid, q_b))
    triplet_seeds.sort(reverse=True)
    top_triplets = triplet_seeds[:3]

    # --- Hardware paths of length 3: p_a - p_mid - p_b ---
    # Prefer central midpoints with high degree (routing hubs in heavy-hex)
    central_phys = sorted(physical_qubits, key=lambda p: (cent[p], -phys_degree[p]))[:50]
    hw_paths_3 = []
    for p_mid in central_phys:
        nbs = list(self.backend[p_mid])
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                p_a, p_b = nbs[i], nbs[j]
                score = cent[p_mid] + cent[p_a] + cent[p_b]
                hw_paths_3.append((score, p_a, p_mid, p_b))
    hw_paths_3.sort()
    top_hw_paths_3 = [(p_a, p_mid, p_b) for _, p_a, p_mid, p_b in hw_paths_3[:4]]

    if not top_phys_pairs or not top_logic_pairs:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 11 – Evaluate all seeds (greedy fill, then compare cost)      #
    # ------------------------------------------------------------------ #
    rem_phys_pool = sorted(physical_qubits, key=lambda p: phys_degree[p], reverse=True)
    best_cost = float('inf')
    best_lq_to_phys = None

    def evaluate_seed(seed_assignment):
        nonlocal best_cost, best_lq_to_phys
        candidate = greedy_fill(seed_assignment)
        placed_phys_set = set(candidate.values())
        rem_phys = [p for p in rem_phys_pool if p not in placed_phys_set]
        rem_lqs = [lq for lq in logical_qubits if lq not in candidate]
        for lq, phys in zip(rem_lqs, rem_phys):
            candidate[lq] = phys
        c = mapping_cost(candidate)
        if c < best_cost:
            best_cost = c
            best_lq_to_phys = dict(candidate)

    # Pair seeds
    for sq1, sq2 in top_logic_pairs:
        for sp1, sp2 in top_phys_pairs:
            for a, b in [(sq1, sq2), (sq2, sq1)]:
                evaluate_seed({a: sp1, b: sp2})

    # Triplet chain seeds: map logical chain to hardware path
    for _, tq_a, tq_mid, tq_b in top_triplets:
        for tp_a, tp_mid, tp_b in top_hw_paths_3:
            for la, lm, lb in [(tq_a, tq_mid, tq_b), (tq_b, tq_mid, tq_a)]:
                evaluate_seed({la: tp_a, lm: tp_mid, lb: tp_b})

    if best_lq_to_phys is None:
        best_lq_to_phys = {lq: phys for lq, phys in zip(logical_qubits, physical_qubits)}

    # ------------------------------------------------------------------ #
    # Step 12 – Hill-climbing on best seed result                        #
    # ------------------------------------------------------------------ #
    hill_climb(best_lq_to_phys, max_rounds=6)
    best_cost = mapping_cost(best_lq_to_phys)
    # Snapshot of best after initial hill-climbing
    ils_start = dict(best_lq_to_phys)

    # ------------------------------------------------------------------ #
    # Step 13 – Iterated Local Search (ILS)                             #
    # Perturbation: randomly select K logical qubits and shuffle their   #
    # physical assignments. This escapes hill-climbing local optima      #
    # without the overhead of 3-way rotations or simulated annealing.   #
    # ------------------------------------------------------------------ #
    lq_list_ils = logical_qubits[:]
    n_ils = len(lq_list_ils)
    RESTART_COUNT = 14
    # Perturbation size: ~10-15% of qubits, clamped to [4, 18]
    K = max(4, min(18, n_ils // 7))

    for _ in range(RESTART_COUNT):
        # Start each restart from the current global best
        perturbed = dict(best_lq_to_phys)

        # Random segment permutation: pick K qubits, shuffle their physicals
        selected_lqs = rng.sample(lq_list_ils, min(K, n_ils))
        selected_phys = [perturbed[lq] for lq in selected_lqs]
        rng.shuffle(selected_phys)
        for lq, ph in zip(selected_lqs, selected_phys):
            perturbed[lq] = ph

        hill_climb(perturbed, max_rounds=5)
        c = mapping_cost(perturbed)
        if c < best_cost:
            best_cost = c
            best_lq_to_phys = dict(perturbed)

    lq_to_phys = best_lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 14 – Build strict 1-to-1 bijection via in-place swap chain   #
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