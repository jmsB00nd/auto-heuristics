def init_mapping(self):
    """
    Critical-Path Iterated Local Search with Window Weighting (CPILS-WW)

    Improvements over CPMR-3OLS (293.00):

    1. Window-weighted interactions: 2-qubit gates in the first third of
       the circuit get 2x weight; middle third 1.5x; last third 1x.
       Early gates are executed first — their swap costs are paid
       immediately and cascade to all later gates. Complementary to (not
       replacing) critical-path weighting.

    2. Expanded restart budget: TOP_K_EDGES=5 × TOP_K_PAIRS=3 = 15
       edge-based restarts (was 3×2 = 6).

    3. Hub-anchor restarts: map the most-connected logical qubit to the
       top-3 most-central physical qubits, catching star-topology
       sub-circuits that edge-based anchors miss.

    4. Iterated Local Search (ILS) escape: after multi-restart converges,
       run 8 ILS iterations. Each iteration:
         (a) ranks logical qubits by contribution to total cost,
         (b) applies a deterministic cyclic-rotation perturbation on the
             top-K worst contributors (K grows 3→10 across iterations),
         (c) re-runs pairwise + 3-way local search,
         (d) accepts if better; otherwise restarts from global best.
       This escapes local optima unreachable by exhaustive pairwise/3-way.
    """
    import heapq
    from collections import defaultdict

    # ------------------------------------------------------------------ #
    # Logical / physical qubit sets                                       #
    # ------------------------------------------------------------------ #
    logical_qubit_set = set()
    for qubits in self.access.values():
        for q in qubits:
            logical_qubit_set.add(q)

    logical_qubits  = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict         = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # Step 1 – DAG critical-path analysis                                 #
    # ------------------------------------------------------------------ #
    sorted_gates = sorted(self.access.keys())
    n_gates      = len(sorted_gates)

    qubit_last_gate = {}
    gate_prev       = [[] for _ in range(n_gates)]

    for i, gate in enumerate(sorted_gates):
        for q in self.access[gate]:
            if q in qubit_last_gate:
                gate_prev[i].append(qubit_last_gate[q])
            qubit_last_gate[q] = i

    fwd_depth = [0] * n_gates
    for i in range(n_gates):
        if gate_prev[i]:
            fwd_depth[i] = 1 + max(fwd_depth[p] for p in gate_prev[i])

    max_fwd = max(fwd_depth) if fwd_depth else 1

    gate_succ = [[] for _ in range(n_gates)]
    for i in range(n_gates):
        for p in gate_prev[i]:
            gate_succ[p].append(i)

    bwd_depth = [0] * n_gates
    for i in range(n_gates - 1, -1, -1):
        if gate_succ[i]:
            bwd_depth[i] = 1 + max(bwd_depth[s] for s in gate_succ[i])

    # ------------------------------------------------------------------ #
    # Step 2 – Interaction weights: critical-path × window boost         #
    #   window_boost: 2.0 (first third), 1.5 (middle), 1.0 (last third) #
    #   cp_boost: 1.0 + 0.5*fwd_norm + 0.5*bwd_norm                     #
    # ------------------------------------------------------------------ #
    iw = defaultdict(lambda: defaultdict(float))
    for i, gate in enumerate(sorted_gates):
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2  = qubits[0], qubits[1]
            rel_pos = i / max(n_gates - 1, 1)
            if rel_pos < 1.0 / 3.0:
                window_boost = 2.0
            elif rel_pos < 2.0 / 3.0:
                window_boost = 1.5
            else:
                window_boost = 1.0
            fwd_norm = fwd_depth[i] / (max_fwd + 1)
            bwd_norm = bwd_depth[i] / (max_fwd + 1)
            combined = window_boost * (1.0 + 0.5 * fwd_norm + 0.5 * bwd_norm)
            iw[q1][q2] += combined
            iw[q2][q1] += combined

    # ------------------------------------------------------------------ #
    # Step 3 – Hardware structures                                        #
    # ------------------------------------------------------------------ #
    phys_degree  = {p: len(self.backend[p]) for p in physical_qubits}
    phys_adj_set = {p: set(self.backend[p]) for p in physical_qubits}

    def _mean_dist(p):
        vals = [self.distance_matrix[p][o] for o in physical_qubits
                if o != p and self.distance_matrix[p][o] != float('inf')]
        return sum(vals) / len(vals) if vals else float('inf')

    phys_centrality = {p: _mean_dist(p) for p in physical_qubits}

    # ------------------------------------------------------------------ #
    # Total weighted-distance cost (objective to minimise)               #
    # ------------------------------------------------------------------ #
    def _total_cost(mapping):
        cost = 0.0
        for q1 in logical_qubits:
            if q1 not in mapping:
                continue
            for q2, w in iw[q1].items():
                if q2 > q1 and q2 in mapping:
                    d = self.distance_matrix[mapping[q1]][mapping[q2]]
                    if d != float('inf'):
                        cost += w * d
        return cost

    # ------------------------------------------------------------------ #
    # Greedy BFS placement given initial anchor assignments               #
    # ------------------------------------------------------------------ #
    def _greedy_place(anchor_assignments):
        lq_to_phys  = dict(anchor_assignments)
        placed_phys = set(lq_to_phys.values())
        heap    = []
        counter = [0]

        def _push_unplaced(lq):
            for nb, w in iw[lq].items():
                if nb not in lq_to_phys:
                    heapq.heappush(heap, (-w, counter[0], nb))
                    counter[0] += 1

        for placed_lq in list(lq_to_phys.keys()):
            _push_unplaced(placed_lq)

        if not heap:
            for lq in logical_qubits:
                if lq not in lq_to_phys:
                    heapq.heappush(heap, (0.0, counter[0], lq))
                    counter[0] += 1
                    break

        while heap:
            _, _, lq = heapq.heappop(heap)
            if lq in lq_to_phys:
                continue
            available_phys = [p for p in physical_qubits if p not in placed_phys]
            if not available_phys:
                break

            best_phys  = None
            best_score = None

            for p in available_phys:
                direct = 0.0
                for nb_lq, w in iw[lq].items():
                    if nb_lq in lq_to_phys:
                        d = self.distance_matrix[p][lq_to_phys[nb_lq]]
                        if d != float('inf'):
                            direct += w / (d + 1)

                lookahead  = 0.0
                avail_excl = [q for q in available_phys if q != p]
                if avail_excl:
                    for nb_lq, w in iw[lq].items():
                        if nb_lq not in lq_to_phys:
                            best_d = min(
                                (self.distance_matrix[p][q] for q in avail_excl
                                 if self.distance_matrix[p][q] != float('inf')),
                                default=float('inf')
                            )
                            if best_d != float('inf'):
                                lookahead += w * 0.4 / (best_d + 1)

                score = (direct + lookahead, phys_degree[p])
                if best_score is None or score > best_score:
                    best_score = score
                    best_phys  = p

            if best_phys is None:
                break

            lq_to_phys[lq] = best_phys
            placed_phys.add(best_phys)
            _push_unplaced(lq)

        remaining_phys = sorted(
            [p for p in physical_qubits if p not in placed_phys],
            key=lambda p: phys_degree[p], reverse=True
        )
        for lq, phys in zip(
            [lq for lq in logical_qubits if lq not in lq_to_phys],
            remaining_phys
        ):
            lq_to_phys[lq] = phys

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # Local search: pairwise swaps + 3-way cyclic permutations            #
    # ------------------------------------------------------------------ #
    def _local_search(lq_to_phys, max_iters=20):
        placed_lqs = [lq for lq in logical_qubits if lq in lq_to_phys]
        lq_total_w = sorted(placed_lqs,
                            key=lambda lq: sum(iw[lq].values()), reverse=True)
        top_lqs    = lq_total_w[:min(20, len(lq_total_w))]

        for _ in range(max_iters):
            improved = False

            # Pairwise swaps
            for i in range(len(placed_lqs)):
                for j in range(i + 1, len(placed_lqs)):
                    lq1, lq2 = placed_lqs[i], placed_lqs[j]
                    p1, p2   = lq_to_phys[lq1], lq_to_phys[lq2]
                    delta = 0.0
                    for nb, w in iw[lq1].items():
                        if nb in lq_to_phys and nb != lq2:
                            nb_p = lq_to_phys[nb]
                            d_b  = self.distance_matrix[p1][nb_p]
                            d_a  = self.distance_matrix[p2][nb_p]
                            if d_b != float('inf') and d_a != float('inf'):
                                delta += w * (d_a - d_b)
                    for nb, w in iw[lq2].items():
                        if nb in lq_to_phys and nb != lq1:
                            nb_p = lq_to_phys[nb]
                            d_b  = self.distance_matrix[p2][nb_p]
                            d_a  = self.distance_matrix[p1][nb_p]
                            if d_b != float('inf') and d_a != float('inf'):
                                delta += w * (d_a - d_b)
                    if delta < -1e-9:
                        lq_to_phys[lq1] = p2
                        lq_to_phys[lq2] = p1
                        improved = True

            # 3-way cyclic permutations (when pairwise stalls)
            if not improved:
                for i in range(len(top_lqs)):
                    for j in range(i + 1, len(top_lqs)):
                        for k in range(j + 1, len(top_lqs)):
                            lq1, lq2, lq3 = top_lqs[i], top_lqs[j], top_lqs[k]
                            p1 = lq_to_phys[lq1]
                            p2 = lq_to_phys[lq2]
                            p3 = lq_to_phys[lq3]
                            triple  = {lq1, lq2, lq3}
                            old_map = {lq1: p1, lq2: p2, lq3: p3}
                            for (np1, np2, np3) in [(p2, p3, p1), (p3, p1, p2)]:
                                new_map = {lq1: np1, lq2: np2, lq3: np3}
                                delta   = 0.0
                                seen    = set()
                                for lq in triple:
                                    op  = old_map[lq]
                                    npv = new_map[lq]
                                    for nb, w in iw[lq].items():
                                        edge = (min(lq, nb), max(lq, nb))
                                        if edge in seen or nb not in lq_to_phys:
                                            continue
                                        seen.add(edge)
                                        nb_old = old_map[nb] if nb in triple else lq_to_phys[nb]
                                        nb_new = new_map[nb] if nb in triple else lq_to_phys[nb]
                                        d_bef  = self.distance_matrix[op][nb_old]
                                        d_aft  = self.distance_matrix[npv][nb_new]
                                        if d_bef != float('inf') and d_aft != float('inf'):
                                            delta += w * (d_aft - d_bef)
                                if delta < -1e-9:
                                    lq_to_phys[lq1] = np1
                                    lq_to_phys[lq2] = np2
                                    lq_to_phys[lq3] = np3
                                    improved = True
                                    break

            if not improved:
                break

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # Step 4 – Build restart candidates                                   #
    # ------------------------------------------------------------------ #
    edge_weights = []
    for q1 in logical_qubits:
        for q2, w in iw[q1].items():
            if q2 > q1:
                nbw   = sum(iw[q1].values()) + sum(iw[q2].values()) - 2 * w
                score = w + 0.15 * nbw
                edge_weights.append((score, q1, q2))
    edge_weights.sort(reverse=True)

    TOP_K_EDGES = min(5, len(edge_weights)) if edge_weights else 0

    adjacent_pairs = []
    for p1 in physical_qubits:
        for p2 in phys_adj_set[p1]:
            if p2 > p1:
                score = (phys_centrality[p1] + phys_centrality[p2],
                         -(phys_degree[p1] + phys_degree[p2]))
                adjacent_pairs.append((score, p1, p2))
    adjacent_pairs.sort()
    TOP_K_PAIRS    = min(3, len(adjacent_pairs))
    top_phys_pairs = [(p1, p2) for _, p1, p2 in adjacent_pairs[:TOP_K_PAIRS]]

    best_cost    = float('inf')
    best_mapping = None

    def _run_and_track(anchor):
        nonlocal best_cost, best_mapping
        placed = _greedy_place(anchor)
        placed = _local_search(placed)
        cost   = _total_cost(placed)
        if cost < best_cost:
            best_cost    = cost
            best_mapping = dict(placed)

    # Edge × physical-pair restarts
    if edge_weights:
        for _, alq1, alq2 in edge_weights[:TOP_K_EDGES]:
            lq1_w = sum(iw[alq1].values())
            lq2_w = sum(iw[alq2].values())
            for bp1, bp2 in top_phys_pairs:
                if lq1_w >= lq2_w:
                    orientations = [(bp1, bp2) if phys_degree[bp1] >= phys_degree[bp2]
                                    else (bp2, bp1)]
                else:
                    orientations = [(bp2, bp1) if phys_degree[bp1] >= phys_degree[bp2]
                                    else (bp1, bp2)]
                other = (orientations[0][1], orientations[0][0])
                if other not in orientations:
                    orientations.append(other)
                for pp1, pp2 in orientations:
                    _run_and_track({alq1: pp1, alq2: pp2})

    # Hub-anchor restarts: most-connected logical → top-3 central physical
    if logical_qubits:
        hub_lq   = max(logical_qubits, key=lambda lq: sum(iw[lq].values()))
        hub_phys = sorted(physical_qubits,
                          key=lambda p: (phys_centrality[p], -phys_degree[p]))
        for phys_hub in hub_phys[:3]:
            _run_and_track({hub_lq: phys_hub})

    if best_mapping is None:
        anchor_phys  = min(physical_qubits, key=lambda p: phys_centrality[p])
        placed       = _greedy_place({logical_qubits[0]: anchor_phys})
        placed       = _local_search(placed)
        best_mapping = placed
        best_cost    = _total_cost(best_mapping)

    # ------------------------------------------------------------------ #
    # Step 5 – Iterated Local Search (ILS) escape                        #
    # ------------------------------------------------------------------ #
    # Each iteration: perturb worst-contributing qubits via cyclic
    # rotation (deterministic, no randomness), then re-run local search.
    # K grows from 3 to 10 across iterations for diversity.
    N_ILS_ITERS     = 8
    current_mapping = dict(best_mapping)
    current_cost    = best_cost

    for ils_iter in range(N_ILS_ITERS):
        # Compute per-qubit cost contribution
        qubit_contrib = {}
        for lq in logical_qubits:
            if lq not in current_mapping:
                continue
            contrib = 0.0
            for nb, w in iw[lq].items():
                if nb > lq and nb in current_mapping:
                    d = self.distance_matrix[current_mapping[lq]][current_mapping[nb]]
                    if d != float('inf'):
                        contrib += w * d
            qubit_contrib[lq] = contrib

        sorted_by_contrib = sorted(qubit_contrib,
                                   key=lambda lq: qubit_contrib[lq], reverse=True)
        K = min(3 + ils_iter, len(sorted_by_contrib))
        if K < 2:
            break

        # Cyclic rotation: lq[i] gets lq[(i+1)%K]'s physical position
        perturb_lqs = sorted_by_contrib[:K]
        perturbed   = dict(current_mapping)
        phys_cycle  = [current_mapping[lq] for lq in perturb_lqs]
        for idx, lq in enumerate(perturb_lqs):
            perturbed[lq] = phys_cycle[(idx + 1) % K]

        perturbed = _local_search(perturbed, max_iters=15)
        cost      = _total_cost(perturbed)

        if cost < best_cost:
            best_cost    = cost
            best_mapping = dict(perturbed)

        if cost <= current_cost:
            current_mapping = dict(perturbed)
            current_cost    = cost
        else:
            # Restart from global best to avoid drifting
            current_mapping = dict(best_mapping)
            current_cost    = best_cost

    lq_to_phys = best_mapping

    # ------------------------------------------------------------------ #
    # Step 6 – Strict 1-to-1 permutation via in-place swap               #
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