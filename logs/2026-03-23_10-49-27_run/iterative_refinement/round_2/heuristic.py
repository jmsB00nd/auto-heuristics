def init_mapping(self):
    """
    Slack-Aware Temporal Mapping with 3-opt Local Search (SATM-3LS)

    Key improvements over CSIWM-LS (Round 1, 297.57 mean swaps):
      1. Circuit layering: topological depth level assigned to each gate.
         Earlier layers get a temporal weight boost (TEMPORAL_BOOST=1.5),
         since gates that execute first impose routing constraints first.
      2. Dual-pass critical-path analysis: forward + backward depth gives
         slack per gate. qubit_criticality[q] = max over all gates g
         containing q of 1/(1+slack[g]).  This is a continuous score in
         (0,1] rather than the binary on/off of the previous round, so
         the graduated bonus avoids a hard cliff at the path boundary.
      3. 3-opt cyclic refinement (for n ≤ 60) after 2-opt: can escape
         local optima unreachable by any pairwise swap.
      4. More local-search passes (30 vs 20).
    """
    from collections import defaultdict, deque

    # ------------------------------------------------------------------ #
    # 0. Collect logical / physical qubit sets                            #
    # ------------------------------------------------------------------ #
    logical_qubit_set = set()
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
    # 1. Assign circuit-depth levels (topological ordering)               #
    # ------------------------------------------------------------------ #
    sorted_gates = sorted(self.access.keys())
    last_level_on_qubit = {}
    gate_level = {}

    for gate in sorted_gates:
        qubits = self.access[gate]
        level = max(
            (last_level_on_qubit.get(q, -1) for q in qubits),
            default=-1
        ) + 1
        gate_level[gate] = level
        for q in qubits:
            last_level_on_qubit[q] = level

    max_level = max(gate_level.values()) if gate_level else 0

    # ------------------------------------------------------------------ #
    # 2. Temporally-weighted interaction graph                            #
    #    w(gate) = 1 + BOOST * (1 - level / max_level)                   #
    #    level-0 gates: weight = 1 + BOOST; last-level: weight = 1       #
    # ------------------------------------------------------------------ #
    TEMPORAL_BOOST = 1.5
    interaction_neighbors = defaultdict(lambda: defaultdict(float))

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            lvl = gate_level.get(gate, 0)
            w = 1.0 + TEMPORAL_BOOST * (1.0 - lvl / max(max_level, 1))
            interaction_neighbors[q1][q2] += w
            interaction_neighbors[q2][q1] += w

    # ------------------------------------------------------------------ #
    # 3. Dual-pass critical-path analysis → continuous slack score        #
    # ------------------------------------------------------------------ #
    two_qubit_gates = {g: q for g, q in self.access.items() if len(q) == 2}
    qubit_criticality = defaultdict(float)

    if two_qubit_gates:
        sorted_2q = sorted(two_qubit_gates.keys())
        last_gate_on_q = {}
        dag_succ = defaultdict(set)
        dag_pred = defaultdict(set)

        for gate in sorted_2q:
            q1, q2 = two_qubit_gates[gate]
            for q in (q1, q2):
                if q in last_gate_on_q:
                    pred = last_gate_on_q[q]
                    dag_succ[pred].add(gate)
                    dag_pred[gate].add(pred)
            last_gate_on_q[q1] = gate
            last_gate_on_q[q2] = gate

        # Forward pass
        in_deg = {g: len(dag_pred[g]) for g in sorted_2q}
        fwd = {g: 1 for g in sorted_2q}
        topo_order = []
        queue = deque(g for g in sorted_2q if in_deg[g] == 0)

        while queue:
            node = queue.popleft()
            topo_order.append(node)
            for succ in dag_succ[node]:
                if fwd[node] + 1 > fwd[succ]:
                    fwd[succ] = fwd[node] + 1
                in_deg[succ] -= 1
                if in_deg[succ] == 0:
                    queue.append(succ)

        # Backward pass
        bwd = {g: 1 for g in sorted_2q}
        for node in reversed(topo_order):
            for succ in dag_succ[node]:
                if bwd[succ] + 1 > bwd[node]:
                    bwd[node] = bwd[succ] + 1

        crit_len = max(fwd[g] + bwd[g] - 1 for g in sorted_2q) if sorted_2q else 1

        # Continuous criticality: 1.0 for zero-slack, decays with slack
        for g in sorted_2q:
            slack = crit_len - (fwd[g] + bwd[g] - 1)
            crit_score = 1.0 / (1.0 + slack)
            for q in two_qubit_gates[g]:
                qubit_criticality[q] = max(qubit_criticality[q], crit_score)

    # ------------------------------------------------------------------ #
    # 4. Enhanced weights: graduated criticality bonus                    #
    #    bonus = 1 + CRIT_SCALE * min(crit[q1], crit[q2])               #
    #    For zero-slack pairs: bonus = 1 + CRIT_SCALE (≈5×)             #
    # ------------------------------------------------------------------ #
    CRIT_SCALE = 4.0
    enhanced = defaultdict(dict)
    for q1 in interaction_neighbors:
        for q2, w in interaction_neighbors[q1].items():
            c1 = qubit_criticality.get(q1, 0.0)
            c2 = qubit_criticality.get(q2, 0.0)
            bonus = 1.0 + CRIT_SCALE * min(c1, c2)
            enhanced[q1][q2] = w * bonus

    # ------------------------------------------------------------------ #
    # 5. Hardware centroid seed                                            #
    # ------------------------------------------------------------------ #
    center_phys = min(
        physical_qubits,
        key=lambda p: sum(
            self.distance_matrix[p][o]
            for o in physical_qubits
            if self.distance_matrix[p][o] != float('inf')
        )
    )

    anchor_lq = max(
        logical_qubits,
        key=lambda q: sum(enhanced[q].values()) if enhanced[q] else 0
    )
    lq_to_phys = {anchor_lq: center_phys}
    placed_phys = {center_phys}
    unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]

    # ------------------------------------------------------------------ #
    # 6. Greedy BFS expansion with enhanced weights                       #
    # ------------------------------------------------------------------ #
    while unplaced:
        next_lq = max(
            unplaced,
            key=lambda lq: sum(enhanced[lq].get(p_lq, 0) for p_lq in lq_to_phys)
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
            total = 0.0
            for p_lq, p_phys in lq_to_phys.items():
                w = enhanced[_lq].get(p_lq, 0)
                if w > 0:
                    d = self.distance_matrix[phys_c][p_phys]
                    total += w * (d if d != float('inf') else 1e9)
            return total

        best_phys = min(candidates, key=placement_cost)
        lq_to_phys[next_lq] = best_phys
        placed_phys.add(best_phys)
        unplaced.remove(next_lq)

    # ------------------------------------------------------------------ #
    # 7. Local search: 2-opt (all sizes) + 3-opt cycles (n ≤ 60)        #
    # ------------------------------------------------------------------ #
    lq_list = [lq for lq in logical_qubits if lq in lq_to_phys]
    n = len(lq_list)
    use_3opt = (n <= 60)

    for _pass in range(30):
        improved = False

        # --- 2-opt: pairwise swap ---
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

        # --- 3-opt: cyclic rotation of triples ---
        if use_3opt:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        lq1 = lq_list[i]
                        lq2 = lq_list[j]
                        lq3 = lq_list[k]
                        p1 = lq_to_phys[lq1]
                        p2 = lq_to_phys[lq2]
                        p3 = lq_to_phys[lq3]
                        triple_set = {lq1, lq2, lq3}

                        for (np1, np2, np3) in [(p2, p3, p1), (p3, p1, p2)]:
                            delta = 0.0
                            old_pos = {lq1: p1, lq2: p2, lq3: p3}
                            new_pos = {lq1: np1, lq2: np2, lq3: np3}

                            # Edges from each triple member to outside nodes
                            for lq_a in (lq1, lq2, lq3):
                                for lq_b, w in enhanced[lq_a].items():
                                    if lq_b in triple_set or lq_b not in lq_to_phys:
                                        continue
                                    p_b = lq_to_phys[lq_b]
                                    d_old = self.distance_matrix[old_pos[lq_a]][p_b]
                                    d_new = self.distance_matrix[new_pos[lq_a]][p_b]
                                    delta += w * (
                                        (d_new if d_new != float('inf') else 1e9) -
                                        (d_old if d_old != float('inf') else 1e9)
                                    )

                            # Edges within the triple (each pair counted once)
                            for (lq_a, lq_b) in [(lq1, lq2), (lq1, lq3), (lq2, lq3)]:
                                w = enhanced[lq_a].get(lq_b, 0.0)
                                if w > 0:
                                    d_old = self.distance_matrix[old_pos[lq_a]][old_pos[lq_b]]
                                    d_new = self.distance_matrix[new_pos[lq_a]][new_pos[lq_b]]
                                    delta += w * (
                                        (d_new if d_new != float('inf') else 1e9) -
                                        (d_old if d_old != float('inf') else 1e9)
                                    )

                            if delta < -1e-9:
                                lq_to_phys[lq1] = np1
                                lq_to_phys[lq2] = np2
                                lq_to_phys[lq3] = np3
                                improved = True
                                break  # Don't try the other rotation for this triple

        if not improved:
            break

    # ------------------------------------------------------------------ #
    # 8. Build strict bijection via in-place swap                         #
    # ------------------------------------------------------------------ #
    mapping_dict         = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq]                   = target_phys
        mapping_dict[displaced_lq]         = current_phys
        reverse_mapping_dict[target_phys]  = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)