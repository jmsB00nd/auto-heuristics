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
    # Step 2 – Topological sort + gate layers                            #
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
    # Step 5 – Balanced position-criticality interaction weights         #
    # NEW: w(g) = (cp+1) * (max_layer - layer + 1)                      #
    # Product form naturally rewards gates that are BOTH early AND       #
    # on long critical paths without aggressive exponential decay.       #
    # Also tracks "urgency" = first 30% of circuit for tie-breaking.    #
    # ------------------------------------------------------------------ #
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    early_thresh = max(1, int(max_layer * 0.30))

    interaction_neighbors = defaultdict(dict)
    urgency = defaultdict(dict)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        cp = critical_path[g] + 1
        layer = gate_layer[g]
        w = cp * (max_layer - layer + 1)
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0.0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0.0) + w
        if layer <= early_thresh:
            wu = cp * (early_thresh - layer + 1)
            urgency[q1][q2] = urgency[q1].get(q2, 0.0) + wu
            urgency[q2][q1] = urgency[q2].get(q1, 0.0) + wu

    weighted_degree = {q: sum(interaction_neighbors[q].values()) for q in logical_qubits}
    urgency_degree  = {q: sum(urgency[q].values())             for q in logical_qubits}

    # ------------------------------------------------------------------ #
    # Step 6 – Hardware structures                                        #
    # ------------------------------------------------------------------ #
    def _centrality(p):
        finite = [self.distance_matrix[p][o] for o in physical_qubits
                  if o != p and self.distance_matrix[p][o] != float('inf')]
        return sum(finite) / len(finite) if finite else float('inf')

    cent       = {p: _centrality(p)       for p in physical_qubits}
    phys_degree = {p: len(self.backend[p]) for p in physical_qubits}
    phys_by_cent = sorted(physical_qubits, key=lambda p: cent[p])

    # ------------------------------------------------------------------ #
    # Step 7 – Mapping cost                                               #
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
    # Step 8 – Greedy BFS-frontier fill with urgency tie-breaking        #
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
                    urgency_degree.get(lq, 0.0),
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
    # Step 9 – Hill climbing: pairwise + 3-way cyclic (small circuits)  #
    # 3-way cyclic permutations escape local optima that pairwise swaps  #
    # cannot reach (e.g., a->b->c->a rotation with net improvement).    #
    # ------------------------------------------------------------------ #
    def hill_climb(lq_to_phys, max_rounds=8, do_triple=True):
        lq_list = logical_qubits[:]
        n_lq = len(lq_list)

        # Phase 1: pairwise swaps until convergence
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

        # Phase 2: 3-way cyclic permutations (only for small circuits)
        if do_triple and n_lq <= 35:
            triple_improved = False
            for i in range(n_lq):
                lq1 = lq_list[i]
                p1 = lq_to_phys.get(lq1)
                if p1 is None:
                    continue
                for j in range(i + 1, n_lq):
                    lq2 = lq_list[j]
                    w12 = interaction_neighbors[lq1].get(lq2, 0.0)
                    p2 = lq_to_phys.get(lq2)
                    if p2 is None:
                        continue
                    for k in range(j + 1, n_lq):
                        lq3 = lq_list[k]
                        w13 = interaction_neighbors[lq1].get(lq3, 0.0)
                        w23 = interaction_neighbors[lq2].get(lq3, 0.0)
                        if w12 == 0.0 and w13 == 0.0 and w23 == 0.0:
                            continue
                        p3 = lq_to_phys.get(lq3)
                        if p3 is None:
                            continue

                        # Precompute pairwise physical distances
                        d12 = self.distance_matrix[p1][p2]; d12 = d12 if d12 != float('inf') else 1e9
                        d13 = self.distance_matrix[p1][p3]; d13 = d13 if d13 != float('inf') else 1e9
                        d23 = self.distance_matrix[p2][p3]; d23 = d23 if d23 != float('inf') else 1e9

                        # Inner triangle cost for each permutation
                        # Current: lq1->p1, lq2->p2, lq3->p3
                        cur_in = w12 * d12 + w13 * d13 + w23 * d23
                        # Perm A: lq1->p2, lq2->p3, lq3->p1
                        pA_in  = w12 * d23 + w13 * d12 + w23 * d13
                        # Perm B: lq1->p3, lq2->p1, lq3->p2
                        pB_in  = w12 * d13 + w13 * d23 + w23 * d12

                        # External cost per permutation
                        ext_cur = ext_A = ext_B = 0.0
                        for other_lq, other_phys in lq_to_phys.items():
                            if other_lq in (lq1, lq2, lq3):
                                continue
                            wa = interaction_neighbors[lq1].get(other_lq, 0.0)
                            wb = interaction_neighbors[lq2].get(other_lq, 0.0)
                            wc = interaction_neighbors[lq3].get(other_lq, 0.0)
                            if wa + wb + wc == 0.0:
                                continue
                            da = self.distance_matrix[p1][other_phys]; da = da if da != float('inf') else 1e9
                            db = self.distance_matrix[p2][other_phys]; db = db if db != float('inf') else 1e9
                            dc = self.distance_matrix[p3][other_phys]; dc = dc if dc != float('inf') else 1e9
                            ext_cur += wa * da + wb * db + wc * dc
                            ext_A   += wa * db + wb * dc + wc * da
                            ext_B   += wa * dc + wb * da + wc * db

                        cur_tot = cur_in + ext_cur
                        pA_tot  = pA_in  + ext_A
                        pB_tot  = pB_in  + ext_B
                        best_tot = min(pA_tot, pB_tot)

                        if best_tot < cur_tot - 1e-9:
                            if pA_tot < pB_tot:
                                lq_to_phys[lq1] = p2
                                lq_to_phys[lq2] = p3
                                lq_to_phys[lq3] = p1
                            else:
                                lq_to_phys[lq1] = p3
                                lq_to_phys[lq2] = p1
                                lq_to_phys[lq3] = p2
                            triple_improved = True

            # Phase 3: cleanup pairwise after 3-way moves
            if triple_improved:
                for _ in range(4):
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
    # Step 10 – Spectral seed via approximate Fiedler vector             #
    # Power iteration on random-walk Laplacian gives 1-D ordering of     #
    # logical qubits by graph structure; map to HW by centrality order.  #
    # ------------------------------------------------------------------ #
    def spectral_ordering(adj, nodes, n_iter=20):
        n = len(nodes)
        if n <= 2:
            return nodes[:]
        node_idx = {v: i for i, v in enumerate(nodes)}
        deg = [sum(adj[v].values()) for v in nodes]

        v = [rng.gauss(0, 1) for _ in range(n)]
        mean = sum(v) / n
        v = [x - mean for x in v]
        norm = math.sqrt(sum(x * x for x in v))
        if norm < 1e-10:
            return nodes[:]
        v = [x / norm for x in v]

        for _ in range(n_iter):
            Lv = [0.0] * n
            for vi_idx, node in enumerate(nodes):
                d = deg[vi_idx]
                if d == 0:
                    Lv[vi_idx] = 0.0
                    continue
                nb_sum = sum(
                    adj[node].get(nb, 0.0) * v[node_idx[nb]]
                    for nb in adj[node] if nb in node_idx
                )
                Lv[vi_idx] = v[vi_idx] - nb_sum / d
            mean = sum(Lv) / n
            Lv = [x - mean for x in Lv]
            norm = math.sqrt(sum(x * x for x in Lv))
            if norm < 1e-10:
                break
            v = [x / norm for x in Lv]

        return [nodes[i] for i in sorted(range(n), key=lambda x: v[x])]

    # ------------------------------------------------------------------ #
    # Step 11 – Build diverse seeds                                       #
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

    triplet_seeds = []
    for q_mid in logical_qubits:
        nbs = list(interaction_neighbors[q_mid].items())
        for i in range(len(nbs)):
            for j in range(i + 1, len(nbs)):
                q_a, w_a = nbs[i]
                q_b, w_b = nbs[j]
                w_ab = interaction_neighbors[q_a].get(q_b, 0.0)
                score = w_a + w_b + 0.3 * w_ab
                triplet_seeds.append((score, q_a, q_mid, q_b))
    triplet_seeds.sort(reverse=True)
    top_triplets = triplet_seeds[:4]

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
    top_hw_paths_3 = [(p_a, p_mid, p_b) for _, p_a, p_mid, p_b in hw_paths_3[:5]]

    top_hub_lqs  = sorted(logical_qubits, key=lambda q: weighted_degree.get(q, 0), reverse=True)[:3]
    top_hub_phys = sorted(physical_qubits, key=lambda p: (cent[p], -phys_degree[p]))[:5]

    if not top_phys_pairs or not top_logic_pairs:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 12 – Evaluate all seeds                                        #
    # ------------------------------------------------------------------ #
    rem_phys_pool = sorted(physical_qubits, key=lambda p: phys_degree[p], reverse=True)
    best_cost = float('inf')
    best_lq_to_phys = None

    def evaluate_seed(seed_assignment):
        nonlocal best_cost, best_lq_to_phys
        candidate = greedy_fill(seed_assignment)
        placed_phys_set = set(candidate.values())
        rem_phys = [p for p in rem_phys_pool if p not in placed_phys_set]
        rem_lqs  = [lq for lq in logical_qubits if lq not in candidate]
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

    # Triplet chain seeds
    for _, tq_a, tq_mid, tq_b in top_triplets:
        for tp_a, tp_mid, tp_b in top_hw_paths_3:
            for la, lm, lb in [(tq_a, tq_mid, tq_b), (tq_b, tq_mid, tq_a)]:
                evaluate_seed({la: tp_a, lm: tp_mid, lb: tp_b})

    # Hub/star seeds
    for hub_lq in top_hub_lqs:
        for hub_phys in top_hub_phys:
            evaluate_seed({hub_lq: hub_phys})

    # Spectral seed: Fiedler-ordered logical -> centrality-ordered physical
    try:
        fiedler_order = spectral_ordering(interaction_neighbors, logical_qubits)
        n_map = min(len(fiedler_order), len(phys_by_cent))
        evaluate_seed({fiedler_order[i]: phys_by_cent[i] for i in range(n_map)})
        evaluate_seed({fiedler_order[i]: phys_by_cent[n_map - 1 - i] for i in range(n_map)})
    except Exception:
        pass

    if best_lq_to_phys is None:
        best_lq_to_phys = {lq: phys for lq, phys in zip(logical_qubits, physical_qubits)}

    # ------------------------------------------------------------------ #
    # Step 13 – Initial hill climb on best seed result                   #
    # ------------------------------------------------------------------ #
    n_lq_total = len(logical_qubits)
    hill_climb(best_lq_to_phys, max_rounds=8, do_triple=(n_lq_total <= 35))
    best_cost = mapping_cost(best_lq_to_phys)

    # ------------------------------------------------------------------ #
    # Step 14 – ILS with LNS + Simulated Annealing acceptance            #
    #                                                                     #
    # SA acceptance allows the restart point to occasionally accept      #
    # slightly worse solutions, enabling escape from deep basins.        #
    # Global best is always tracked separately.                          #
    # ------------------------------------------------------------------ #
    lq_list_ils = logical_qubits[:]
    n_ils = len(lq_list_ils)

    K_small = max(3, min(10, n_ils // 10))
    K_large = max(8, min(24, n_ils // 5))
    K_rand  = max(4, min(18, n_ils // 7))

    def lns_worst_subgraph(m, K):
        worst_cost_val = -1.0
        worst_pair = None
        for lq1 in lq_list_ils:
            p1 = m.get(lq1)
            if p1 is None:
                continue
            for lq2, w in interaction_neighbors[lq1].items():
                if lq1 < lq2 and lq2 in m:
                    d = self.distance_matrix[p1][m[lq2]]
                    c = w * (d if d != float('inf') else 1e9)
                    if c > worst_cost_val:
                        worst_cost_val = c
                        worst_pair = (lq1, lq2)

        if worst_pair is None:
            return rng.sample(lq_list_ils, min(K, n_ils))

        removed = list(worst_pair)
        in_set = set(worst_pair)
        frontier = list(worst_pair)
        while len(in_set) < K and frontier:
            next_frontier = []
            for lq in frontier:
                sorted_nbs = sorted(
                    interaction_neighbors[lq].keys(),
                    key=lambda x: interaction_neighbors[lq][x],
                    reverse=True,
                )
                for nb in sorted_nbs:
                    if nb not in in_set and nb in m:
                        in_set.add(nb)
                        removed.append(nb)
                        next_frontier.append(nb)
                        if len(in_set) >= K:
                            break
                if len(in_set) >= K:
                    break
            frontier = next_frontier
        return removed[:K]

    TOTAL_RESTARTS = 22
    LNS_RESTARTS   = 14

    # SA: allow accepting up to 2% worse solutions early, cooling to 0
    T_init = 0.02 * best_cost if best_cost > 0 else 1.0
    sa_best_cost    = best_cost
    sa_best_mapping = dict(best_lq_to_phys)

    for restart_idx in range(TOTAL_RESTARTS):
        perturbed = dict(sa_best_mapping)
        T = T_init * (1.0 - restart_idx / TOTAL_RESTARTS)

        if restart_idx < LNS_RESTARTS:
            K = K_small if restart_idx % 2 == 0 else K_large
            to_remove_set = set(lns_worst_subgraph(perturbed, K))
            partial = {lq: ph for lq, ph in perturbed.items() if lq not in to_remove_set}
            perturbed = greedy_fill(partial)
            placed_set = set(perturbed.values())
            rem = [p for p in rem_phys_pool if p not in placed_set]
            for lq in lq_list_ils:
                if lq not in perturbed:
                    if rem:
                        perturbed[lq] = rem.pop(0)
        else:
            selected_lqs  = rng.sample(lq_list_ils, min(K_rand, n_ils))
            selected_phys = [perturbed[lq] for lq in selected_lqs]
            rng.shuffle(selected_phys)
            for lq, ph in zip(selected_lqs, selected_phys):
                perturbed[lq] = ph

        hill_climb(perturbed, max_rounds=6, do_triple=False)
        c = mapping_cost(perturbed)

        # Update global best
        if c < best_cost:
            best_cost       = c
            best_lq_to_phys = dict(perturbed)

        # SA acceptance for restart point (allows exploring worse solutions)
        delta = c - sa_best_cost
        if delta < 0 or (T > 0 and rng.random() < math.exp(-delta / T)):
            sa_best_cost    = c
            sa_best_mapping = dict(perturbed)

    # Final cleanup hill climb on global best
    hill_climb(best_lq_to_phys, max_rounds=8, do_triple=(n_ils <= 35))

    lq_to_phys = best_lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 15 – Build strict 1-to-1 bijection via in-place swap chain   #
    # ------------------------------------------------------------------ #
    mapping_dict         = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq]           = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys]   = lq
        reverse_mapping_dict[current_phys]  = displaced_lq

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)