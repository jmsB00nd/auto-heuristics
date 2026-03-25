def init_mapping(self):
    """
    ACAM-ILS: Adaptive Chain-Anchored Mapping with Iterated Local Search

    Improvements over Round 7 (TSMM-LG, score 282.38):
    1. Critical chain hardware path embedding: detect the primary interaction
       chain from the DAG critical path and seed placements from hardware
       path alignments — structurally superior to single-anchor greedy for
       circuits with long sequential interactions.
    2. 3-cycle rotation local search: cyclic permutations among high-
       interaction qubit triples catch improvements pairwise swaps provably
       cannot (3-cycles are not decomposable into improving 2-swaps).
    3. Iterated Local Search (ILS) with double-bridge perturbation: perturb
       the best solution and re-optimize, escaping local optima that
       independent multi-start misses by building on good solutions.
    """
    from collections import defaultdict, deque
    import math
    import random

    # ------------------------------------------------------------------ #
    # 1. Build interaction graphs (raw count + time-weighted)              #
    # ------------------------------------------------------------------ #
    logical_qubit_set = set()
    raw_inter = defaultdict(dict)
    time_inter = defaultdict(dict)

    sorted_gates = sorted(self.access.keys())
    two_q_ordered = [(g, self.access[g]) for g in sorted_gates if len(self.access[g]) == 2]
    N2 = len(two_q_ordered)

    for idx, (gate, qubits) in enumerate(two_q_ordered):
        q1, q2 = qubits[0], qubits[1]
        logical_qubit_set.update([q1, q2])
        raw_inter[q1][q2] = raw_inter[q1].get(q2, 0) + 1
        raw_inter[q2][q1] = raw_inter[q2].get(q1, 0) + 1
        tw = math.exp(-3.0 * idx / max(1, N2 - 1))
        time_inter[q1][q2] = time_inter[q1].get(q2, 0.0) + tw
        time_inter[q2][q1] = time_inter[q2].get(q1, 0.0) + tw

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # 2. Criticality via DP on 2Q gate DAG + critical chain extraction     #
    # ------------------------------------------------------------------ #
    two_qubit_gates = {g: q for g, q in self.access.items() if len(q) == 2}
    qubit_criticality = defaultdict(float)
    dp_len = {}
    dp_prev_map = {}
    critical_chain_qubits = []  # ordered qubit sequence along critical path

    if two_qubit_gates:
        sorted_2q = sorted(two_qubit_gates.keys())
        last_g = {}
        dag_succ = defaultdict(set)
        dag_pred = defaultdict(set)

        for gate in sorted_2q:
            q1, q2 = two_qubit_gates[gate]
            for q in (q1, q2):
                if q in last_g:
                    pred = last_g[q]
                    dag_succ[pred].add(gate)
                    dag_pred[gate].add(pred)
            last_g[q1] = gate
            last_g[q2] = gate

        in_deg = {g: len(dag_pred[g]) for g in sorted_2q}
        dp_len = {g: 1 for g in sorted_2q}
        dp_prev_map = {g: None for g in sorted_2q}
        q_topo = deque(g for g in sorted_2q if in_deg[g] == 0)

        while q_topo:
            node = q_topo.popleft()
            for succ in dag_succ[node]:
                if dp_len[node] + 1 > dp_len[succ]:
                    dp_len[succ] = dp_len[node] + 1
                    dp_prev_map[succ] = node
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    q_topo.append(succ)

        cp_len = max(dp_len.values()) if dp_len else 1
        threshold = max(1, cp_len * 0.82)

        for g in sorted_2q:
            if dp_len[g] >= threshold:
                r = dp_len[g] / cp_len
                for q in two_qubit_gates[g]:
                    qubit_criticality[q] = max(qubit_criticality[q], r)

        # Trace back critical path and collect ordered qubit sequence
        node = max(sorted_2q, key=lambda g: dp_len[g])
        cp_path = []
        while node is not None:
            for q in two_qubit_gates[node]:
                qubit_criticality[q] = 1.0
            cp_path.append(node)
            node = dp_prev_map[node]
        cp_path.reverse()

        seen_q = set()
        for gate in cp_path:
            for q in two_qubit_gates[gate]:
                if q not in seen_q:
                    critical_chain_qubits.append(q)
                    seen_q.add(q)

    # ------------------------------------------------------------------ #
    # 3. Enhanced weights: time-blend + criticality bonus                  #
    # ------------------------------------------------------------------ #
    MAX_BONUS = 5.0
    TIME_BLEND = 0.40

    enhanced = defaultdict(dict)
    for q1 in raw_inter:
        for q2, w_raw in raw_inter[q1].items():
            c1 = qubit_criticality.get(q1, 0.0)
            c2 = qubit_criticality.get(q2, 0.0)
            bonus = 1.0 + (MAX_BONUS - 1.0) * (c1 + c2) * 0.5
            w_time = time_inter[q1].get(q2, 0.0)
            enhanced[q1][q2] = ((1 - TIME_BLEND) * w_raw + TIME_BLEND * w_time) * bonus

    # ------------------------------------------------------------------ #
    # 4. Hardware centrality + path finding (NEW: for chain seeding)       #
    # ------------------------------------------------------------------ #
    hw_sum_dist = {
        p: sum(
            self.distance_matrix[p][o]
            for o in physical_qubits
            if self.distance_matrix[p][o] != float('inf')
        )
        for p in physical_qubits
    }

    def find_hw_paths(length, max_paths=20):
        """Find hardware paths of given length using limited DFS."""
        if length <= 1:
            return [[p] for p in physical_qubits[:max_paths]]
        paths = []
        # Start DFS from most-central physical qubits
        central_phys = sorted(physical_qubits, key=lambda p: hw_sum_dist[p])

        def dfs(path, visited):
            if len(paths) >= max_paths:
                return
            if len(path) == length:
                paths.append(path[:])
                return
            for nb in self.backend[path[-1]]:
                if nb not in visited:
                    path.append(nb)
                    visited.add(nb)
                    dfs(path, visited)
                    path.pop()
                    visited.discard(nb)

        for start in central_phys[:20]:
            if len(paths) >= max_paths:
                break
            dfs([start], {start})

        return paths

    # ------------------------------------------------------------------ #
    # 5. Objective function                                                 #
    # ------------------------------------------------------------------ #
    def total_cost(lq_to_phys):
        total = 0.0
        seen = set()
        for q1, q2_dict in enhanced.items():
            if q1 not in lq_to_phys:
                continue
            for q2, w in q2_dict.items():
                if q2 not in lq_to_phys:
                    continue
                pair = (min(q1, q2), max(q1, q2))
                if pair in seen:
                    continue
                seen.add(pair)
                d = self.distance_matrix[lq_to_phys[q1]][lq_to_phys[q2]]
                total += w * (d if d != float('inf') else 1e9)
        return total

    # ------------------------------------------------------------------ #
    # 6. Greedy expansion with balanced ordering + 1-step lookahead        #
    # ------------------------------------------------------------------ #
    def greedy_expand(initial_map=None, anchor_lq=None, anchor_phys=None):
        if initial_map is not None:
            lq_to_phys = dict(initial_map)
        else:
            lq_to_phys = {anchor_lq: anchor_phys}

        placed_phys = set(lq_to_phys.values())
        unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]
        unplaced_set = set(unplaced)

        while unplaced:
            next_lq = max(
                unplaced,
                key=lambda lq: (
                    sum(enhanced[lq].get(pl, 0) for pl in lq_to_phys)
                    + 0.15 * sum(enhanced[lq].get(u, 0) for u in unplaced_set if u != lq)
                )
            )

            cands = list({
                nb for phys in placed_phys
                for nb in self.backend[phys] if nb not in placed_phys
            })
            if not cands:
                cands = [p for p in physical_qubits if p not in placed_phys]
            if not cands:
                break

            top_unplaced_nb = max(
                (lq for lq in unplaced if lq != next_lq and enhanced[next_lq].get(lq, 0) > 0),
                key=lambda lq: enhanced[next_lq].get(lq, 0),
                default=None
            )

            def place_cost(phys_c):
                cost = sum(
                    enhanced[next_lq].get(pl, 0) * (
                        self.distance_matrix[phys_c][pp]
                        if self.distance_matrix[phys_c][pp] != float('inf') else 1e9
                    )
                    for pl, pp in lq_to_phys.items()
                    if enhanced[next_lq].get(pl, 0) > 0
                )
                if top_unplaced_nb is None:
                    return cost

                temp_placed = placed_phys | {phys_c}
                nb_cands = [
                    nb for phys in temp_placed
                    for nb in self.backend[phys] if nb not in temp_placed
                ][:12]
                if not nb_cands:
                    nb_cands = [p for p in physical_qubits if p not in temp_placed][:12]
                if not nb_cands:
                    return cost

                temp_map_items = list(lq_to_phys.items()) + [(next_lq, phys_c)]
                best_nb_cost = min(
                    sum(
                        enhanced[top_unplaced_nb].get(pl, 0) * (
                            self.distance_matrix[nc][pp]
                            if self.distance_matrix[nc][pp] != float('inf') else 1e9
                        )
                        for pl, pp in temp_map_items
                        if enhanced[top_unplaced_nb].get(pl, 0) > 0
                    )
                    for nc in nb_cands
                )
                return cost + 0.25 * best_nb_cost

            best_phys = min(cands, key=place_cost)
            lq_to_phys[next_lq] = best_phys
            placed_phys.add(best_phys)
            unplaced.remove(next_lq)
            unplaced_set.discard(next_lq)

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # 7. Enhanced local search: swap + relocate + 3-cycle (NEW)            #
    # ------------------------------------------------------------------ #
    def local_search(lq_to_phys, max_passes=30):
        lq_list = [lq for lq in logical_qubits if lq in lq_to_phys]
        occupied = set(lq_to_phys.values())
        unoccupied = [p for p in physical_qubits if p not in occupied]
        n = len(lq_list)

        # Precompute top qubits for 3-cycle search (by interaction degree)
        top_lqs = sorted(lq_list, key=lambda lq: -sum(enhanced[lq].values()))[:min(n, 16)]

        for _pass in range(max_passes):
            improved = False

            # --- Pairwise swap moves ---
            for i in range(n):
                for j in range(i + 1, n):
                    lq1, lq2 = lq_list[i], lq_list[j]
                    p1, p2 = lq_to_phys[lq1], lq_to_phys[lq2]
                    delta = 0.0

                    for lq3, w in enhanced[lq1].items():
                        if lq3 not in lq_to_phys or lq3 == lq2:
                            continue
                        p3 = lq_to_phys[lq3]
                        d_old = self.distance_matrix[p1][p3]
                        d_new = self.distance_matrix[p2][p3]
                        delta += w * (
                            (d_new if d_new != float('inf') else 1e9) -
                            (d_old if d_old != float('inf') else 1e9)
                        )

                    for lq3, w in enhanced[lq2].items():
                        if lq3 not in lq_to_phys or lq3 == lq1:
                            continue
                        p3 = lq_to_phys[lq3]
                        d_old = self.distance_matrix[p2][p3]
                        d_new = self.distance_matrix[p1][p3]
                        delta += w * (
                            (d_new if d_new != float('inf') else 1e9) -
                            (d_old if d_old != float('inf') else 1e9)
                        )

                    if delta < -1e-9:
                        lq_to_phys[lq1], lq_to_phys[lq2] = p2, p1
                        improved = True

            # --- Relocation moves ---
            if unoccupied:
                for lq in lq_list:
                    p_old = lq_to_phys[lq]
                    best_delta = -1e-9
                    best_p = None

                    for p_new in unoccupied:
                        delta = 0.0
                        for lq2, w in enhanced[lq].items():
                            if lq2 not in lq_to_phys:
                                continue
                            p2 = lq_to_phys[lq2]
                            d_old = self.distance_matrix[p_old][p2]
                            d_new = self.distance_matrix[p_new][p2]
                            delta += w * (
                                (d_new if d_new != float('inf') else 1e9) -
                                (d_old if d_old != float('inf') else 1e9)
                            )
                        if delta < best_delta:
                            best_delta = delta
                            best_p = p_new

                    if best_p is not None:
                        lq_to_phys[lq] = best_p
                        unoccupied.remove(best_p)
                        unoccupied.append(p_old)
                        improved = True

            # --- 3-cycle rotation moves among top high-interaction qubits (NEW) ---
            # Cyclic rotations catch improvements unreachable via pairwise swaps.
            m = len(top_lqs)
            if m >= 3:
                for i in range(m):
                    for j in range(i + 1, m):
                        for k in range(j + 1, m):
                            lq1 = top_lqs[i]
                            lq2 = top_lqs[j]
                            lq3 = top_lqs[k]

                            w12 = enhanced[lq1].get(lq2, 0)
                            w13 = enhanced[lq1].get(lq3, 0)
                            w23 = enhanced[lq2].get(lq3, 0)
                            if w12 + w13 + w23 == 0:
                                continue

                            p1 = lq_to_phys[lq1]
                            p2 = lq_to_phys[lq2]
                            p3 = lq_to_phys[lq3]

                            def cycle_delta(a1, a2, a3):
                                """Cost delta for assigning lq1→a1, lq2→a2, lq3→a3."""
                                d = 0.0
                                new_pos = {lq1: a1, lq2: a2, lq3: a3}
                                old_pos = {lq1: p1, lq2: p2, lq3: p3}

                                # Internal pair interactions
                                for lqa, lqb, w in [(lq1, lq2, w12),
                                                    (lq1, lq3, w13),
                                                    (lq2, lq3, w23)]:
                                    if w == 0:
                                        continue
                                    do = self.distance_matrix[old_pos[lqa]][old_pos[lqb]]
                                    dn = self.distance_matrix[new_pos[lqa]][new_pos[lqb]]
                                    d += w * ((dn if dn != float('inf') else 1e9) -
                                              (do if do != float('inf') else 1e9))

                                # External interactions
                                for lq_in in (lq1, lq2, lq3):
                                    for lq_ext, w_ext in enhanced[lq_in].items():
                                        if lq_ext in (lq1, lq2, lq3):
                                            continue
                                        if lq_ext not in lq_to_phys:
                                            continue
                                        p_ext = lq_to_phys[lq_ext]
                                        do = self.distance_matrix[old_pos[lq_in]][p_ext]
                                        dn = self.distance_matrix[new_pos[lq_in]][p_ext]
                                        d += w_ext * (
                                            (dn if dn != float('inf') else 1e9) -
                                            (do if do != float('inf') else 1e9)
                                        )
                                return d

                            # Try both non-trivial cyclic rotations
                            d_rot1 = cycle_delta(p2, p3, p1)  # lq1→p2, lq2→p3, lq3→p1
                            d_rot2 = cycle_delta(p3, p1, p2)  # lq1→p3, lq2→p1, lq3→p2

                            if d_rot1 < -1e-9 and d_rot1 <= d_rot2:
                                lq_to_phys[lq1] = p2
                                lq_to_phys[lq2] = p3
                                lq_to_phys[lq3] = p1
                                improved = True
                            elif d_rot2 < -1e-9:
                                lq_to_phys[lq1] = p3
                                lq_to_phys[lq2] = p1
                                lq_to_phys[lq3] = p2
                                improved = True

            if not improved:
                break

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # 8. Seed selection                                                     #
    # ------------------------------------------------------------------ #
    K_PHYS = min(4, len(physical_qubits))
    top_phys_seeds = sorted(physical_qubits, key=lambda p: hw_sum_dist[p])[:K_PHYS]

    enhanced_deg = {lq: sum(enhanced[lq].values()) for lq in logical_qubits}
    K_LQ = min(4, len(logical_qubits))
    top_lq_seeds = sorted(logical_qubits, key=lambda lq: -enhanced_deg[lq])[:K_LQ]

    best_mapping = None
    best_score = float('inf')

    # ------------------------------------------------------------------ #
    # 9. Multi-start: 4x4 anchor seeds                                     #
    # ------------------------------------------------------------------ #
    for anchor_lq in top_lq_seeds:
        for anchor_phys in top_phys_seeds:
            lq_to_phys = greedy_expand(anchor_lq=anchor_lq, anchor_phys=anchor_phys)
            lq_to_phys = local_search(lq_to_phys)
            score = total_cost(lq_to_phys)
            if score < best_score:
                best_score = score
                best_mapping = dict(lq_to_phys)

    # ------------------------------------------------------------------ #
    # 10. Chain-anchored seeds: embed critical chain on hardware paths (NEW)#
    # ------------------------------------------------------------------ #
    if len(critical_chain_qubits) >= 3:
        chain_len = min(len(critical_chain_qubits), 7)
        lq_chain = critical_chain_qubits[:chain_len]

        hw_paths = find_hw_paths(chain_len, max_paths=20)

        if hw_paths:
            def score_chain_path(hw_path):
                s = 0.0
                for ci in range(len(lq_chain)):
                    for cj in range(ci + 1, len(lq_chain)):
                        w = enhanced.get(lq_chain[ci], {}).get(lq_chain[cj], 0.0)
                        if w > 0:
                            d = self.distance_matrix[hw_path[ci]][hw_path[cj]]
                            s += w * (d if d != float('inf') else 1e9)
                return s

            top_chain_paths = sorted(hw_paths, key=score_chain_path)[:4]

            for hw_path in top_chain_paths:
                chain_map = {lq_chain[i]: hw_path[i] for i in range(len(lq_chain))}
                if len(set(chain_map.values())) < len(chain_map):
                    continue  # skip degenerate paths
                lq_to_phys = greedy_expand(initial_map=chain_map)
                lq_to_phys = local_search(lq_to_phys)
                score = total_cost(lq_to_phys)
                if score < best_score:
                    best_score = score
                    best_mapping = dict(lq_to_phys)

    # ------------------------------------------------------------------ #
    # 11. ILS: Iterated Local Search with double-bridge perturbation (NEW) #
    # ------------------------------------------------------------------ #
    rng = random.Random(42)
    ils_current = dict(best_mapping)
    ils_current_score = best_score

    ILS_ITERS = 20

    for ils_it in range(ILS_ITERS):
        perturbed = dict(ils_current)
        lqs = [lq for lq in logical_qubits if lq in perturbed]

        # Alternating perturbation strength: 2 or 3 pair swaps
        strength = 2 + (ils_it % 2)
        if len(lqs) >= 2 * strength:
            idxs = rng.sample(range(len(lqs)), 2 * strength)
            for ki in range(0, len(idxs) - 1, 2):
                lq_a, lq_b = lqs[idxs[ki]], lqs[idxs[ki + 1]]
                perturbed[lq_a], perturbed[lq_b] = perturbed[lq_b], perturbed[lq_a]

        perturbed = local_search(perturbed, max_passes=20)
        p_score = total_cost(perturbed)

        if p_score < best_score:
            best_score = p_score
            best_mapping = dict(perturbed)

        # Accept-if-better ILS baseline (helps diversify search)
        if p_score < ils_current_score:
            ils_current = perturbed
            ils_current_score = p_score

    lq_to_phys = best_mapping

    # ------------------------------------------------------------------ #
    # 12. Build strict 1-to-1 bijection via in-place swap                  #
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