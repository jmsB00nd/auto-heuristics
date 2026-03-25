def init_mapping(self):
    """
    ILS-3R: Iterated Local Search with 3-way Rotation Moves (Round 8)

    Key improvements over Round 7 (TSMM-LG, score 282.38):
    1. Iterated Local Search (ILS): After local search convergence per seed,
       apply random perturbation + re-run local search 2 times to escape
       local optima that multi-start alone cannot reach.
    2. 3-way rotation moves in local search: Extend the search neighborhood
       beyond pairwise swaps to cyclic permutations of 3 interacting qubits.
       These moves can escape local optima unreachable by any pair swap.
    3. 2-hop physical neighbor expansion in greedy: Use neighbors within
       distance 2 on the hardware graph as placement candidates, giving the
       greedy phase access to a richer set of physical positions.
    4. Interaction-sorted pair processing: Check high-interaction pairs first
       in local search for faster convergence per pass.
    """
    from collections import defaultdict, deque
    import math
    import random

    # -------------------------------------------------------------- #
    # 1. Build interaction graphs (raw count + time-weighted)         #
    # -------------------------------------------------------------- #
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

    # -------------------------------------------------------------- #
    # 2. DAG-based qubit criticality                                  #
    # -------------------------------------------------------------- #
    two_qubit_gates = {g: q for g, q in self.access.items() if len(q) == 2}
    qubit_criticality = defaultdict(float)
    dp_len = {}
    dp_prev_map = {}

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

        node = max(sorted_2q, key=lambda g: dp_len[g])
        while node is not None:
            for q in two_qubit_gates[node]:
                qubit_criticality[q] = 1.0
            node = dp_prev_map[node]

    # -------------------------------------------------------------- #
    # 3. Enhanced interaction weights                                  #
    # -------------------------------------------------------------- #
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

    # -------------------------------------------------------------- #
    # 4. Precompute 2-hop physical neighbors                          #
    # -------------------------------------------------------------- #
    phys_2hop = {}
    for p in physical_qubits:
        hop1 = set(self.backend[p])
        hop2 = set()
        for nb in hop1:
            hop2.update(self.backend[nb])
        phys_2hop[p] = hop1 | hop2

    # -------------------------------------------------------------- #
    # 5. Seed selection                                               #
    # -------------------------------------------------------------- #
    hw_sum_dist = {
        p: sum(
            self.distance_matrix[p][o]
            for o in physical_qubits
            if self.distance_matrix[p][o] != float('inf')
        )
        for p in physical_qubits
    }
    K_PHYS = min(4, len(physical_qubits))
    top_phys_seeds = sorted(physical_qubits, key=lambda p: hw_sum_dist[p])[:K_PHYS]

    enhanced_deg = {lq: sum(enhanced[lq].values()) for lq in logical_qubits}
    K_LQ = min(4, len(logical_qubits))
    top_lq_seeds = sorted(logical_qubits, key=lambda lq: -enhanced_deg[lq])[:K_LQ]

    # -------------------------------------------------------------- #
    # 6. Objective function                                           #
    # -------------------------------------------------------------- #
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

    # -------------------------------------------------------------- #
    # 7. Greedy expansion with 2-hop candidates + 1-step lookahead   #
    # -------------------------------------------------------------- #
    def greedy_expand(anchor_lq, anchor_phys):
        lq_to_phys = {anchor_lq: anchor_phys}
        placed_phys = {anchor_phys}
        unplaced = [lq for lq in logical_qubits if lq != anchor_lq]
        unplaced_set = set(unplaced)

        while unplaced:
            next_lq = max(
                unplaced,
                key=lambda lq: (
                    sum(enhanced[lq].get(pl, 0) for pl in lq_to_phys)
                    + 0.15 * sum(enhanced[lq].get(u, 0) for u in unplaced_set if u != lq)
                )
            )

            # NEW: use 2-hop neighbors for richer candidate set
            cands = list({
                nb for phys in placed_phys
                for nb in phys_2hop[phys] if nb not in placed_phys
            })
            if not cands:
                cands = [p for p in physical_qubits if p not in placed_phys]
            if not cands:
                break

            # Limit candidates by proximity to placed region
            if len(cands) > 20:
                cands = sorted(cands, key=lambda p: min(
                    (self.distance_matrix[p][pp] for pp in placed_phys
                     if self.distance_matrix[p][pp] != float('inf')),
                    default=1e9
                ))[:20]

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

    # -------------------------------------------------------------- #
    # 8. Local search: pairwise swaps + 3-way rotations + relocation  #
    # -------------------------------------------------------------- #
    def local_search(lq_to_phys, max_passes=30):
        lq_list = [lq for lq in logical_qubits if lq in lq_to_phys]
        occupied = set(lq_to_phys.values())
        unoccupied = [p for p in physical_qubits if p not in occupied]
        n = len(lq_list)

        # NEW: sort pairs by interaction weight — focus on high-weight pairs
        pair_weights = []
        for i in range(n):
            for j in range(i + 1, n):
                w = enhanced[lq_list[i]].get(lq_list[j], 0.0)
                pair_weights.append((i, j, w))
        pair_weights.sort(key=lambda x: -x[2])
        ordered_pairs = [(x[0], x[1]) for x in pair_weights]
        high_pairs = [(x[0], x[1]) for x in pair_weights if x[2] > 0]

        for _pass in range(max_passes):
            improved = False

            # --- Pairwise swaps (interaction-sorted order) ---
            for i, j in ordered_pairs:
                lq1, lq2 = lq_list[i], lq_list[j]
                p1, p2 = lq_to_phys[lq1], lq_to_phys[lq2]
                delta = 0.0

                for lq3, w in enhanced[lq1].items():
                    if lq3 not in lq_to_phys or lq3 == lq2:
                        continue
                    p3 = lq_to_phys[lq3]
                    d_old = self.distance_matrix[p1][p3]
                    d_new = self.distance_matrix[p2][p3]
                    delta += w * ((d_new if d_new != float('inf') else 1e9) -
                                  (d_old if d_old != float('inf') else 1e9))

                for lq3, w in enhanced[lq2].items():
                    if lq3 not in lq_to_phys or lq3 == lq1:
                        continue
                    p3 = lq_to_phys[lq3]
                    d_old = self.distance_matrix[p2][p3]
                    d_new = self.distance_matrix[p1][p3]
                    delta += w * ((d_new if d_new != float('inf') else 1e9) -
                                  (d_old if d_old != float('inf') else 1e9))

                if delta < -1e-9:
                    lq_to_phys[lq1], lq_to_phys[lq2] = p2, p1
                    improved = True

            # --- NEW: 3-way rotation moves ---
            # For top high-interaction pairs (a,b), try cyclic rotations with
            # a third qubit c that interacts with a or b.
            # Rotation 1: a→pos(b), b→pos(c), c→pos(a)
            # Rotation 2: a→pos(c), b→pos(a), c→pos(b)
            for i, j in high_pairs[:min(40, len(high_pairs))]:
                lq_a = lq_list[i]
                lq_b = lq_list[j]

                third_cands = []
                seen_thirds = set()
                for lq_c, wc in enhanced[lq_a].items():
                    if lq_c in lq_to_phys and lq_c != lq_b and wc > 0:
                        third_cands.append(lq_c)
                        seen_thirds.add(lq_c)
                for lq_c, wc in enhanced[lq_b].items():
                    if lq_c in lq_to_phys and lq_c != lq_a and wc > 0 and lq_c not in seen_thirds:
                        third_cands.append(lq_c)

                for lq_c in third_cands[:6]:
                    # Always refresh positions in case previous rotation updated them
                    pa = lq_to_phys[lq_a]
                    pb = lq_to_phys[lq_b]
                    pc = lq_to_phys[lq_c]
                    triple = {lq_a, lq_b, lq_c}
                    new_pos_r1 = {lq_a: pb, lq_b: pc, lq_c: pa}
                    new_pos_r2 = {lq_a: pc, lq_b: pa, lq_c: pb}

                    def rotation_delta(new_pos):
                        d = 0.0
                        # Internal triangle edges
                        for qa, qb_inner in [(lq_a, lq_b), (lq_a, lq_c), (lq_b, lq_c)]:
                            w = enhanced[qa].get(qb_inner, 0.0)
                            if w == 0.0:
                                continue
                            p_a_old = lq_to_phys[qa]
                            p_b_old = lq_to_phys[qb_inner]
                            p_a_new = new_pos[qa]
                            p_b_new = new_pos[qb_inner]
                            d_old = self.distance_matrix[p_a_old][p_b_old]
                            d_new = self.distance_matrix[p_a_new][p_b_new]
                            d += w * ((d_new if d_new != float('inf') else 1e9) -
                                      (d_old if d_old != float('inf') else 1e9))
                        # External edges for each moved qubit
                        for lq_moved, p_new_val in new_pos.items():
                            p_old = lq_to_phys[lq_moved]
                            for lq_nb, w in enhanced[lq_moved].items():
                                if lq_nb not in lq_to_phys or lq_nb in triple:
                                    continue
                                p_nb = lq_to_phys[lq_nb]
                                d_old = self.distance_matrix[p_old][p_nb]
                                d_new = self.distance_matrix[p_new_val][p_nb]
                                d += w * ((d_new if d_new != float('inf') else 1e9) -
                                          (d_old if d_old != float('inf') else 1e9))
                        return d

                    d1 = rotation_delta(new_pos_r1)
                    d2 = rotation_delta(new_pos_r2)
                    best_d = min(d1, d2)

                    if best_d < -1e-9:
                        best_rot = new_pos_r1 if d1 <= d2 else new_pos_r2
                        for lq_moved, p_new_val in best_rot.items():
                            lq_to_phys[lq_moved] = p_new_val
                        improved = True

            # --- Relocation moves: move lq to best unoccupied slot ---
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
                            delta += w * ((d_new if d_new != float('inf') else 1e9) -
                                          (d_old if d_old != float('inf') else 1e9))
                        if delta < best_delta:
                            best_delta = delta
                            best_p = p_new

                    if best_p is not None:
                        lq_to_phys[lq] = best_p
                        unoccupied.remove(best_p)
                        unoccupied.append(p_old)
                        improved = True

            if not improved:
                break

        return lq_to_phys

    # -------------------------------------------------------------- #
    # 9. ILS perturbation (double-bridge style random segment swap)   #
    # -------------------------------------------------------------- #
    def perturb(lq_to_phys, rng, strength=4):
        result = dict(lq_to_phys)
        lq_list_local = [lq for lq in logical_qubits if lq in result]
        if len(lq_list_local) < 4:
            return result
        n_swaps = min(strength, len(lq_list_local) // 2)
        idxs = rng.sample(range(len(lq_list_local)), n_swaps * 2)
        for k in range(0, n_swaps * 2, 2):
            lq1 = lq_list_local[idxs[k]]
            lq2 = lq_list_local[idxs[k + 1]]
            result[lq1], result[lq2] = result[lq2], result[lq1]
        return result

    # -------------------------------------------------------------- #
    # 10. Multi-start (4x4=16 seeds) + ILS (2 rounds per seed)       #
    # -------------------------------------------------------------- #
    best_mapping = None
    best_score = float('inf')
    rng = random.Random(42)

    for anchor_lq in top_lq_seeds:
        for anchor_phys in top_phys_seeds:
            lq_to_phys = greedy_expand(anchor_lq, anchor_phys)
            lq_to_phys = local_search(lq_to_phys, max_passes=30)
            score = total_cost(lq_to_phys)
            if score < best_score:
                best_score = score
                best_mapping = dict(lq_to_phys)

            # NEW: ILS — perturb the converged solution and re-optimize
            current = dict(lq_to_phys)
            current_score = score
            for _ils in range(2):
                perturbed = perturb(current, rng, strength=4)
                perturbed = local_search(perturbed, max_passes=20)
                s = total_cost(perturbed)
                if s < best_score:
                    best_score = s
                    best_mapping = dict(perturbed)
                # Accept improvement (strict descent acceptance)
                if s < current_score:
                    current = perturbed
                    current_score = s

    lq_to_phys = best_mapping

    # -------------------------------------------------------------- #
    # 11. Build strict 1-to-1 bijection via in-place swap             #
    # -------------------------------------------------------------- #
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