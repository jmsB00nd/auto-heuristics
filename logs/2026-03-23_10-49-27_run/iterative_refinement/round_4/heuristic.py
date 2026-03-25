def init_mapping(self):
    """
    DAG-Weighted Multi-Seed + 3-Cycle Local Search + ILS-LNS (DWMS-3C)

    Improvements over CSIWM-LS (Round 1):
    1. Full-DAG gate layers + critical remaining depth weight:
         w(g) = (crit_depth[g]+1) * exp(-alpha * gate_layer[g])
       so early gates on deep critical paths dominate placement.
    2. Multi-seed: top-5 interaction pairs × top-5 hardware adjacent pairs,
       top-4 triplet chains, and hub/star seeds (50+ candidates evaluated).
    3. Neighbour-restricted 3-cycle (3-opt) local search: forward rotation
       (lq1→p2, lq2→p3, lq3→p1) and reverse rotation tried for all
       interaction-edge triples. Escapes pairwise-swap local optima.
       O(n·k²) per pass vs O(n²) for pairwise.
    4. ILS with LNS: BFS from costliest edge collects K-qubit worst subgraph,
       removes and re-places with guided greedy for structured escape.
    5. After each ILS step: hill-climb(5) → 3-cycle(3) → hill-climb(2),
       ensuring every candidate is 3-opt optimal before comparison.
    """
    from collections import defaultdict, deque
    import math
    import random

    rng = random.Random(42)
    all_gates        = sorted(self.access.keys())
    two_qubit_gates  = [g for g in all_gates if len(self.access[g]) == 2]

    # ------------------------------------------------------------------ #
    # Step 1 – Full DAG (all gates for accurate circuit depth)            #
    # ------------------------------------------------------------------ #
    last_gate_on_qubit = {}
    successors_dag     = defaultdict(set)
    predecessors_dag   = defaultdict(set)
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
    # Step 2 – Gate layers (Kahn BFS) + critical remaining depth          #
    # ------------------------------------------------------------------ #
    in_degree  = {g: len(predecessors_dag[g]) for g in all_gates}
    gate_layer = {g: 0 for g in all_gates}
    temp_in    = dict(in_degree)
    queue      = deque(g for g in all_gates if in_degree[g] == 0)
    topo_order = []
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in successors_dag[g]:
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            temp_in[s] -= 1
            if temp_in[s] == 0:
                queue.append(s)

    crit_depth = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors_dag[g]:
            if crit_depth[s] + 1 > crit_depth[g]:
                crit_depth[g] = crit_depth[s] + 1

    # ------------------------------------------------------------------ #
    # Step 3 – Logical / physical qubits                                  #
    # ------------------------------------------------------------------ #
    logical_qubit_set = set()
    for qubits in self.access.values():
        logical_qubit_set.update(qubits)
    logical_qubits  = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits or not two_qubit_gates:
        self.mapping_dict         = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 4 – Interaction weights: crit-depth × layer-decay             #
    #   w(g) = (crit_depth[g]+1) × exp(-alpha × gate_layer[g])          #
    #   alpha = log(10)/(max_layer+1) → last-layer gates weigh 10× less  #
    # ------------------------------------------------------------------ #
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    alpha     = math.log(10.0) / (max_layer + 1)

    interaction_neighbors = defaultdict(dict)
    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        w = (crit_depth[g] + 1) * math.exp(-alpha * gate_layer[g])
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0.0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0.0) + w

    weighted_degree = {q: sum(interaction_neighbors[q].values()) for q in logical_qubits}

    # ------------------------------------------------------------------ #
    # Step 5 – Hardware analysis                                          #
    # ------------------------------------------------------------------ #
    def _centrality(p):
        finite = [self.distance_matrix[p][o] for o in physical_qubits
                  if o != p and self.distance_matrix[p][o] != float('inf')]
        return sum(finite) / len(finite) if finite else float('inf')

    cent     = {p: _centrality(p)       for p in physical_qubits}
    phys_deg = {p: len(self.backend[p]) for p in physical_qubits}

    # ------------------------------------------------------------------ #
    # Step 6 – Mapping cost                                               #
    # ------------------------------------------------------------------ #
    def mapping_cost(m):
        lqs   = list(m.keys())
        total = 0.0
        for i in range(len(lqs)):
            for j in range(i + 1, len(lqs)):
                w = interaction_neighbors[lqs[i]].get(lqs[j], 0.0)
                if w > 0.0:
                    d = self.distance_matrix[m[lqs[i]]][m[lqs[j]]]
                    total += w * (d if d != float('inf') else 1e9)
        return total

    # ------------------------------------------------------------------ #
    # Step 7 – Greedy BFS fill from a partial seed assignment             #
    # ------------------------------------------------------------------ #
    def greedy_fill(seed_assignment):
        lq_to_phys  = dict(seed_assignment)
        placed_phys = set(lq_to_phys.values())
        unplaced    = [lq for lq in logical_qubits if lq not in lq_to_phys]

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
                dist_cost = sum(
                    interaction_neighbors[_lq].get(placed_lq, 0.0) * (
                        self.distance_matrix[phys_c][placed_phys_q]
                        if self.distance_matrix[phys_c][placed_phys_q] != float('inf')
                        else 1e9
                    )
                    for placed_lq, placed_phys_q in lq_to_phys.items()
                    if interaction_neighbors[_lq].get(placed_lq, 0.0) > 0.0
                )
                empty_nb = sum(1 for nb in self.backend[phys_c] if nb not in placed_phys)
                return (dist_cost, max(0, _fn - empty_nb), -empty_nb)

            best_phys = min(candidates, key=_score)
            lq_to_phys[next_lq] = best_phys
            placed_phys.add(best_phys)
            unplaced.remove(next_lq)

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 8 – Pairwise (2-opt) hill-climbing                            #
    # ------------------------------------------------------------------ #
    def hill_climb(lq_to_phys, max_rounds=8):
        lq_list = logical_qubits[:]
        n_lq    = len(lq_list)
        for _ in range(max_rounds):
            improved = False
            for i in range(n_lq):
                for j in range(i + 1, n_lq):
                    lq1 = lq_list[i]; lq2 = lq_list[j]
                    p1  = lq_to_phys.get(lq1); p2 = lq_to_phys.get(lq2)
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
    # Step 9 – 3-cycle (3-opt) local search  *** KEY NEW CONTRIBUTION *** #
    #                                                                      #
    # For every triple (lq1, lq2, lq3) sharing at least one interaction   #
    # edge, tests two cyclic rotations:                                    #
    #   Forward : lq1→p2, lq2→p3, lq3→p1                                 #
    #   Reverse : lq1→p3, lq2→p1, lq3→p2                                 #
    # Accepts when Δ cost < 0.  Neighbour-restriction limits search to    #
    # O(n·k²) per pass (k = avg interaction degree), making it tractable. #
    # Uses restart-on-first-improvement for correctness.                  #
    # ------------------------------------------------------------------ #
    def three_cycle_search(lq_to_phys, max_passes=6):
        lq_keys = [lq for lq in logical_qubits if lq in lq_to_phys]

        def d(a, b):
            v = self.distance_matrix[a][b]
            return v if v != float('inf') else 1e9

        for _pass in range(max_passes):
            found = False
            for lq1 in lq_keys:
                if found:
                    break
                p1       = lq_to_phys[lq1]
                lq1_nbrs = [q for q in interaction_neighbors[lq1] if q in lq_to_phys]
                for lq2 in lq1_nbrs:
                    if found:
                        break
                    p2       = lq_to_phys[lq2]
                    lq2_nbrs = [q for q in interaction_neighbors[lq2]
                                if q in lq_to_phys and q != lq1]
                    for lq3 in lq2_nbrs:
                        p3     = lq_to_phys[lq3]
                        triple = {lq1, lq2, lq3}
                        w12    = interaction_neighbors[lq1].get(lq2, 0.0)
                        w23    = interaction_neighbors[lq2].get(lq3, 0.0)
                        w13    = interaction_neighbors[lq1].get(lq3, 0.0)

                        # --- Forward rotation: lq1→p2, lq2→p3, lq3→p1 ---
                        delta_fwd = 0.0
                        for lq, op, np_ in [(lq1,p1,p2),(lq2,p2,p3),(lq3,p3,p1)]:
                            for lqn, w in interaction_neighbors[lq].items():
                                if lqn not in lq_to_phys or lqn in triple:
                                    continue
                                pn = lq_to_phys[lqn]
                                delta_fwd += w * (d(np_, pn) - d(op, pn))
                        # Internal edge deltas (verified analytically)
                        delta_fwd += (
                            w12 * (d(p2, p3) - d(p1, p2)) +
                            w23 * (d(p3, p1) - d(p2, p3)) +
                            w13 * (d(p1, p2) - d(p1, p3))
                        )
                        if delta_fwd < -1e-9:
                            lq_to_phys[lq1] = p2
                            lq_to_phys[lq2] = p3
                            lq_to_phys[lq3] = p1
                            found = True
                            break

                        # --- Reverse rotation: lq1→p3, lq2→p1, lq3→p2 ---
                        delta_rev = 0.0
                        for lq, op, np_ in [(lq1,p1,p3),(lq2,p2,p1),(lq3,p3,p2)]:
                            for lqn, w in interaction_neighbors[lq].items():
                                if lqn not in lq_to_phys or lqn in triple:
                                    continue
                                pn = lq_to_phys[lqn]
                                delta_rev += w * (d(np_, pn) - d(op, pn))
                        delta_rev += (
                            w12 * (d(p3, p1) - d(p1, p2)) +
                            w23 * (d(p1, p2) - d(p2, p3)) +
                            w13 * (d(p3, p2) - d(p1, p3))
                        )
                        if delta_rev < -1e-9:
                            lq_to_phys[lq1] = p3
                            lq_to_phys[lq2] = p1
                            lq_to_phys[lq3] = p2
                            found = True
                            break

            if not found:
                break
        return lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 10 – Diverse seed generation                                   #
    # ------------------------------------------------------------------ #
    # Pair seeds
    pair_weights = {
        (q1, q2): interaction_neighbors[q1][q2]
        for q1 in logical_qubits for q2 in interaction_neighbors[q1] if q1 < q2
    }
    top_logic_pairs = sorted(pair_weights, key=lambda p: -pair_weights[p])[:5]
    if not top_logic_pairs and len(logical_qubits) >= 2:
        top_logic_pairs = [(logical_qubits[0], logical_qubits[1])]

    hw_adj_pairs   = sorted(
        [(cent[p1] + cent[p2], p1, p2)
         for p1 in physical_qubits for p2 in self.backend[p1] if p1 < p2]
    )
    top_phys_pairs = [(p1, p2) for _, p1, p2 in hw_adj_pairs[:5]]

    # Triplet chain seeds
    triplet_seeds = []
    for q_mid in logical_qubits:
        nbs = list(interaction_neighbors[q_mid].items())
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                q_a, w_a = nbs[i]; q_b, w_b = nbs[j]
                w_ab = interaction_neighbors[q_a].get(q_b, 0.0)
                triplet_seeds.append((w_a + w_b + 0.3 * w_ab, q_a, q_mid, q_b))
    triplet_seeds.sort(reverse=True)
    top_triplets = triplet_seeds[:4]

    central_phys = sorted(physical_qubits, key=lambda p: (cent[p], -phys_deg[p]))[:50]
    hw_paths_3   = []
    for p_mid in central_phys:
        nbs = list(self.backend[p_mid])
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                p_a, p_b = nbs[i], nbs[j]
                hw_paths_3.append((cent[p_mid]+cent[p_a]+cent[p_b], p_a, p_mid, p_b))
    hw_paths_3.sort()
    top_hw_paths_3 = [(p_a, p_mid, p_b) for _, p_a, p_mid, p_b in hw_paths_3[:5]]

    # Hub seeds
    top_hub_lqs  = sorted(logical_qubits, key=lambda q: -weighted_degree.get(q, 0))[:3]
    top_hub_phys = sorted(physical_qubits, key=lambda p: (cent[p], -phys_deg[p]))[:5]

    if not top_phys_pairs or not top_logic_pairs:
        self.mapping_dict         = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 11 – Evaluate all seeds, keep the best greedy fill             #
    # ------------------------------------------------------------------ #
    rem_phys_pool   = sorted(physical_qubits, key=lambda p: -phys_deg[p])
    best_cost       = float('inf')
    best_lq_to_phys = None

    def evaluate_seed(seed_assignment):
        nonlocal best_cost, best_lq_to_phys
        candidate  = greedy_fill(seed_assignment)
        placed_set = set(candidate.values())
        rem        = [p for p in rem_phys_pool if p not in placed_set]
        for lq in logical_qubits:
            if lq not in candidate and rem:
                candidate[lq] = rem.pop(0)
        c = mapping_cost(candidate)
        if c < best_cost:
            best_cost       = c
            best_lq_to_phys = dict(candidate)

    for sq1, sq2 in top_logic_pairs:
        for sp1, sp2 in top_phys_pairs:
            for a, b in [(sq1, sq2), (sq2, sq1)]:
                evaluate_seed({a: sp1, b: sp2})

    for _, tq_a, tq_mid, tq_b in top_triplets:
        for tp_a, tp_mid, tp_b in top_hw_paths_3:
            for la, lm, lb in [(tq_a, tq_mid, tq_b), (tq_b, tq_mid, tq_a)]:
                evaluate_seed({la: tp_a, lm: tp_mid, lb: tp_b})

    for hub_lq in top_hub_lqs:
        for hub_phys in top_hub_phys:
            evaluate_seed({hub_lq: hub_phys})

    if best_lq_to_phys is None:
        best_lq_to_phys = {lq: phys for lq, phys in zip(logical_qubits, physical_qubits)}

    # ------------------------------------------------------------------ #
    # Step 12 – Initial 3-opt local search on best seed result            #
    #   hill-climb(8) → 3-cycle(6) → hill-climb(4)                       #
    # ------------------------------------------------------------------ #
    hill_climb(best_lq_to_phys, max_rounds=8)
    three_cycle_search(best_lq_to_phys, max_passes=6)
    hill_climb(best_lq_to_phys, max_rounds=4)
    best_cost = mapping_cost(best_lq_to_phys)

    # ------------------------------------------------------------------ #
    # Step 13 – ILS with LNS perturbation                                 #
    # ------------------------------------------------------------------ #
    lq_list_ils = logical_qubits[:]
    n_ils       = len(lq_list_ils)
    K_small = max(3, min(10, n_ils // 10))
    K_large = max(8, min(24, n_ils //  5))
    K_rand  = max(4, min(18, n_ils //  7))

    def lns_worst_subgraph(m, K):
        """BFS from the most expensive interaction edge to collect K qubits."""
        worst_c = -1.0; worst_pair = None
        for lq1 in lq_list_ils:
            p1 = m.get(lq1)
            if p1 is None:
                continue
            for lq2, w in interaction_neighbors[lq1].items():
                if lq1 < lq2 and lq2 in m:
                    dd = self.distance_matrix[p1][m[lq2]]
                    c  = w * (dd if dd != float('inf') else 1e9)
                    if c > worst_c:
                        worst_c = c; worst_pair = (lq1, lq2)
        if worst_pair is None:
            return rng.sample(lq_list_ils, min(K, n_ils))

        removed  = list(worst_pair)
        in_set   = set(worst_pair)
        frontier = list(worst_pair)
        while len(in_set) < K and frontier:
            nxt = []
            for lq in frontier:
                for nb in sorted(interaction_neighbors[lq],
                                 key=lambda x: -interaction_neighbors[lq][x]):
                    if nb not in in_set and nb in m:
                        in_set.add(nb); removed.append(nb); nxt.append(nb)
                        if len(in_set) >= K:
                            break
                if len(in_set) >= K:
                    break
            frontier = nxt
        return removed[:K]

    TOTAL_RESTARTS = 15
    LNS_RESTARTS   = 10

    for restart_idx in range(TOTAL_RESTARTS):
        perturbed = dict(best_lq_to_phys)

        if restart_idx < LNS_RESTARTS:
            K         = K_small if restart_idx % 2 == 0 else K_large
            to_remove = set(lns_worst_subgraph(perturbed, K))
            partial   = {lq: ph for lq, ph in perturbed.items()
                         if lq not in to_remove}
            perturbed = greedy_fill(partial)
            placed_set = set(perturbed.values())
            rem = [p for p in rem_phys_pool if p not in placed_set]
            for lq in lq_list_ils:
                if lq not in perturbed and rem:
                    perturbed[lq] = rem.pop(0)
        else:
            sel_lqs  = rng.sample(lq_list_ils, min(K_rand, n_ils))
            sel_phys = [perturbed[lq] for lq in sel_lqs]
            rng.shuffle(sel_phys)
            for lq, ph in zip(sel_lqs, sel_phys):
                perturbed[lq] = ph

        # Every candidate reaches a 3-opt local optimum before comparison
        hill_climb(perturbed, max_rounds=5)
        three_cycle_search(perturbed, max_passes=3)
        hill_climb(perturbed, max_rounds=2)
        c = mapping_cost(perturbed)
        if c < best_cost:
            best_cost       = c
            best_lq_to_phys = dict(perturbed)

    lq_to_phys = best_lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 14 – Build strict 1-to-1 bijection via in-place swap chain    #
    # ------------------------------------------------------------------ #
    mapping_dict         = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq                       = reverse_mapping_dict[target_phys]
        mapping_dict[lq]                   = target_phys
        mapping_dict[displaced_lq]         = current_phys
        reverse_mapping_dict[target_phys]  = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)