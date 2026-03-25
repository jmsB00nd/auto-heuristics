def init_mapping(self):
    """
    Temporal-Decay Multi-Seed Mapping with 3-Cycle Local Search (TDMS-3CLS)

    Improvements over prior approach:
      1. Temporal decay: gate interaction weight = 1/(layer+1), so early
         circuit layers dominate the objective (routing cannot adapt before
         these gates execute).
      2. Continuous criticality: qubit importance = accumulated temporal score
         + normalised DAG longest-path depth through that qubit (not binary).
      3. Pair seeding: the two most critical interacting qubits are co-seeded
         onto a central hardware *edge* (adjacent pair), not a single node,
         guaranteeing their relative placement is optimal from the start.
      4. Multi-start: five seeding strategies are attempted; the candidate
         with lowest weighted-distance cost after local search is retained.
      5. 3-cycle local search: after pairwise swaps converge, cyclic 3-qubit
         rotations (A→B→C→A) are tried to escape pairwise local optima.
    """
    from collections import defaultdict, deque

    # ------------------------------------------------------------------ #
    # 1. Build raw interaction graph                                       #
    # ------------------------------------------------------------------ #
    logical_qubit_set = set()
    raw_interaction = defaultdict(dict)

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            w = raw_interaction[q1].get(q2, 0) + 1
            raw_interaction[q1][q2] = w
            raw_interaction[q2][q1] = w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ------------------------------------------------------------------ #
    # 2. Gate layer assignment + temporal-decay interaction weights       #
    # ------------------------------------------------------------------ #
    two_qubit_gates = {g: q for g, q in self.access.items() if len(q) == 2}
    temporal_interaction = defaultdict(dict)
    qubit_temporal_score = defaultdict(float)
    qubit_depth = defaultdict(int)

    if two_qubit_gates:
        sorted_2q = sorted(two_qubit_gates.keys())

        # Layer via qubit-based forward pass
        last_layer_on_qubit = {}
        gate_layer = {}
        for gate in sorted_2q:
            q1, q2 = two_qubit_gates[gate]
            layer = max(
                last_layer_on_qubit.get(q1, -1),
                last_layer_on_qubit.get(q2, -1)
            ) + 1
            gate_layer[gate] = layer
            last_layer_on_qubit[q1] = layer
            last_layer_on_qubit[q2] = layer

        for gate in sorted_2q:
            q1, q2 = two_qubit_gates[gate]
            tw = 1.0 / (gate_layer[gate] + 1)
            temporal_interaction[q1][q2] = temporal_interaction[q1].get(q2, 0.0) + tw
            temporal_interaction[q2][q1] = temporal_interaction[q2].get(q1, 0.0) + tw
            qubit_temporal_score[q1] += tw
            qubit_temporal_score[q2] += tw

        # DAG longest-path depth via topological DP
        last_gate_on_qubit = {}
        dag_succ = defaultdict(set)
        dag_pred = defaultdict(set)
        for gate in sorted_2q:
            q1, q2 = two_qubit_gates[gate]
            for q in (q1, q2):
                if q in last_gate_on_qubit:
                    dag_succ[last_gate_on_qubit[q]].add(gate)
                    dag_pred[gate].add(last_gate_on_qubit[q])
            last_gate_on_qubit[q1] = gate
            last_gate_on_qubit[q2] = gate

        in_deg = {g: len(dag_pred[g]) for g in sorted_2q}
        dp_len = {g: 1 for g in sorted_2q}
        topo_q = deque(g for g in sorted_2q if in_deg[g] == 0)
        while topo_q:
            node = topo_q.popleft()
            for succ in dag_succ[node]:
                if dp_len[node] + 1 > dp_len[succ]:
                    dp_len[succ] = dp_len[node] + 1
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    topo_q.append(succ)

        max_depth = max(dp_len.values()) if dp_len else 1
        for gate, depth in dp_len.items():
            for q in two_qubit_gates[gate]:
                qubit_depth[q] = max(qubit_depth[q], depth)

    # ------------------------------------------------------------------ #
    # 3. Build enhanced interaction weights                               #
    #    = temporal weight * (1 + criticality_boost of both endpoints)   #
    # ------------------------------------------------------------------ #
    max_depth = max(qubit_depth.values()) if qubit_depth else 1
    qubit_criticality = {
        q: qubit_temporal_score.get(q, 0.0) + qubit_depth.get(q, 0) / max_depth
        for q in logical_qubits
    }

    enhanced = defaultdict(dict)
    for q1 in temporal_interaction:
        for q2, tw in temporal_interaction[q1].items():
            boost = 1.0 + qubit_criticality.get(q1, 0) + qubit_criticality.get(q2, 0)
            enhanced[q1][q2] = tw * boost
    # Fallback: include any raw interactions not covered by temporal weights
    for q1 in raw_interaction:
        for q2, w in raw_interaction[q1].items():
            if q2 not in enhanced.get(q1, {}):
                enhanced[q1][q2] = w * 0.1

    # ------------------------------------------------------------------ #
    # 4. Hardware analysis: node centrality + central edges               #
    # ------------------------------------------------------------------ #
    phys_dist_sum = {
        p: sum(
            self.distance_matrix[p][o]
            for o in physical_qubits
            if self.distance_matrix[p][o] != float('inf')
        )
        for p in physical_qubits
    }
    center_phys = min(physical_qubits, key=lambda p: phys_dist_sum[p])

    # Edges sorted by combined endpoint centrality (lower = more central)
    central_edges = sorted([
        (phys_dist_sum[p] + phys_dist_sum[nb], p, nb)
        for p in physical_qubits
        for nb in self.backend[p]
        if nb > p
    ])

    # ------------------------------------------------------------------ #
    # 5. Helper: total weighted-distance cost                             #
    # ------------------------------------------------------------------ #
    def compute_cost(lq_to_phys):
        total = 0.0
        lqs = list(lq_to_phys.keys())
        for i in range(len(lqs)):
            for j in range(i + 1, len(lqs)):
                w = enhanced.get(lqs[i], {}).get(lqs[j], 0)
                if w > 0:
                    d = self.distance_matrix[lq_to_phys[lqs[i]]][lq_to_phys[lqs[j]]]
                    total += w * (d if d != float('inf') else 1e9)
        return total

    # ------------------------------------------------------------------ #
    # 6. Helper: greedy BFS expansion from a seed mapping                 #
    # ------------------------------------------------------------------ #
    def greedy_expand(seed_mapping):
        lq_to_phys = dict(seed_mapping)
        placed_phys = set(lq_to_phys.values())
        unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]

        while unplaced:
            # Pick unplaced qubit with highest interaction weight to placed qubits
            next_lq = max(
                unplaced,
                key=lambda lq: sum(enhanced.get(lq, {}).get(p_lq, 0)
                                   for p_lq in lq_to_phys)
            )
            candidates = list({
                nb
                for phys in placed_phys
                for nb in self.backend[phys]
                if nb not in placed_phys
            })
            if not candidates:
                candidates = [p for p in physical_qubits if p not in placed_phys]
            if not candidates:
                break

            def placement_cost(phys_c, _lq=next_lq):
                return sum(
                    enhanced.get(_lq, {}).get(p_lq, 0) *
                    (self.distance_matrix[phys_c][p_phys]
                     if self.distance_matrix[phys_c][p_phys] != float('inf') else 1e9)
                    for p_lq, p_phys in lq_to_phys.items()
                    if enhanced.get(_lq, {}).get(p_lq, 0) > 0
                )

            lq_to_phys[next_lq] = min(candidates, key=placement_cost)
            placed_phys.add(lq_to_phys[next_lq])
            unplaced.remove(next_lq)

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # 7. Helper: local search — pairwise swaps + 3-cycle rotations        #
    # ------------------------------------------------------------------ #
    def local_search(lq_to_phys, max_passes=25):
        lq_list = [lq for lq in logical_qubits if lq in lq_to_phys]
        n = len(lq_list)

        for _pass in range(max_passes):
            improved = False

            # --- Pairwise swaps ---
            for i in range(n):
                for j in range(i + 1, n):
                    lq1, lq2 = lq_list[i], lq_list[j]
                    p1, p2 = lq_to_phys[lq1], lq_to_phys[lq2]
                    delta = 0.0

                    for lq3, w in enhanced.get(lq1, {}).items():
                        if lq3 not in lq_to_phys or lq3 == lq2:
                            continue
                        p3 = lq_to_phys[lq3]
                        d_old = self.distance_matrix[p1][p3]
                        d_new = self.distance_matrix[p2][p3]
                        delta += w * (
                            (d_new if d_new != float('inf') else 1e9) -
                            (d_old if d_old != float('inf') else 1e9)
                        )
                    for lq3, w in enhanced.get(lq2, {}).items():
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

            # --- 3-cycle rotations: only when pairwise swaps are exhausted ---
            if not improved and n <= 80:
                involved_set = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            lqa = lq_list[i]
                            lqb = lq_list[j]
                            lqc = lq_list[k]
                            pa = lq_to_phys[lqa]
                            pb = lq_to_phys[lqb]
                            pc = lq_to_phys[lqc]
                            involved = {lqa, lqb, lqc}
                            m_before = {lqa: pa, lqb: pb, lqc: pc}

                            for (na, nb, nc) in [(pb, pc, pa), (pc, pa, pb)]:
                                m_after = {lqa: na, lqb: nb, lqc: nc}
                                delta = 0.0

                                # Interactions with qubits outside the triple
                                for lqx in (lqa, lqb, lqc):
                                    for lqy, w in enhanced.get(lqx, {}).items():
                                        if lqy not in lq_to_phys or lqy in involved:
                                            continue
                                        py = lq_to_phys[lqy]
                                        d_old = self.distance_matrix[m_before[lqx]][py]
                                        d_new = self.distance_matrix[m_after[lqx]][py]
                                        delta += w * (
                                            (d_new if d_new != float('inf') else 1e9) -
                                            (d_old if d_old != float('inf') else 1e9)
                                        )

                                # Interactions within the triple
                                for (lqx, lqy) in [(lqa, lqb), (lqa, lqc), (lqb, lqc)]:
                                    w = enhanced.get(lqx, {}).get(lqy, 0)
                                    if w > 0:
                                        d_old = self.distance_matrix[m_before[lqx]][m_before[lqy]]
                                        d_new = self.distance_matrix[m_after[lqx]][m_after[lqy]]
                                        delta += w * (
                                            (d_new if d_new != float('inf') else 1e9) -
                                            (d_old if d_old != float('inf') else 1e9)
                                        )

                                if delta < -1e-9:
                                    lq_to_phys[lqa] = na
                                    lq_to_phys[lqb] = nb
                                    lq_to_phys[lqc] = nc
                                    improved = True
                                    break  # direction loop

                            if improved:
                                break  # k loop
                        if improved:
                            break  # j loop
                    if improved:
                        break  # i loop

            if not improved:
                break

        return lq_to_phys

    # ------------------------------------------------------------------ #
    # 8. Multi-seed: run 5 strategies, retain best after local search     #
    # ------------------------------------------------------------------ #
    best_mapping = None
    best_cost = float('inf')

    def try_seed(seed):
        nonlocal best_mapping, best_cost
        m = greedy_expand(seed)
        m = local_search(m)
        c = compute_cost(m)
        if c < best_cost:
            best_cost = c
            best_mapping = dict(m)

    # Find the highest-weight interacting logical pair
    best_pair_w = -1
    best_lq_pair = None
    for q1 in enhanced:
        for q2, w in enhanced[q1].items():
            if q2 > q1 and w > best_pair_w:
                best_pair_w = w
                best_lq_pair = (q1, q2)

    # Highest overall weighted-degree logical qubit
    anchor_lq = max(
        logical_qubits,
        key=lambda q: sum(enhanced.get(q, {}).values())
    )

    # Strategy A: anchor qubit → hardware centroid
    try_seed({anchor_lq: center_phys})

    # Strategy B: best logical pair → most central hardware edge (both orientations)
    if best_lq_pair and central_edges:
        lq1, lq2 = best_lq_pair
        _, pe1, pe2 = central_edges[0]
        try_seed({lq1: pe1, lq2: pe2})
        try_seed({lq1: pe2, lq2: pe1})

    # Strategy C: best logical pair → centroid + its best neighbour
    if best_lq_pair:
        lq1, lq2 = best_lq_pair
        neighbours = sorted(
            self.backend.get(center_phys, []),
            key=lambda p: phys_dist_sum[p]
        )
        if neighbours:
            best_nb = neighbours[0]
            try_seed({lq1: center_phys, lq2: best_nb})
            try_seed({lq2: center_phys, lq1: best_nb})

    # Strategy D: second most-central hardware edge, best logical pair
    if best_lq_pair and len(central_edges) > 1:
        lq1, lq2 = best_lq_pair
        _, pe1, pe2 = central_edges[1]
        try_seed({lq1: pe1, lq2: pe2})

    if best_mapping is None:
        # Degenerate fallback (no interactions)
        best_mapping = {lq: physical_qubits[i % len(physical_qubits)]
                        for i, lq in enumerate(logical_qubits)}

    # ------------------------------------------------------------------ #
    # 9. Build strict 1-to-1 bijection via in-place swap                 #
    # ------------------------------------------------------------------ #
    mapping_dict         = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in best_mapping.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq           = reverse_mapping_dict[target_phys]
        mapping_dict[lq]                   = target_phys
        mapping_dict[displaced_lq]         = current_phys
        reverse_mapping_dict[target_phys]  = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)