def init_mapping(self):
    num_q = self.num_qubits

    # Step 1: Seed mapping — activity-to-centrality rank assignment
    logical_qubits_active = sorted(
        self.logical_activity.keys(),
        key=lambda q: self.logical_activity[q],
        reverse=True,
    )
    physical_qubits_central = sorted(
        self.physical_centrality.keys(),
        key=lambda q: self.physical_centrality[q],
        reverse=True,
    )

    mapping = list(range(num_q))
    reverse_mapping = list(range(num_q))

    used_physical = set()
    assigned_logical = set()

    for lq, pq in zip(logical_qubits_active, physical_qubits_central):
        if lq < num_q and pq < num_q:
            mapping[lq] = pq
            reverse_mapping[pq] = lq
            used_physical.add(pq)
            assigned_logical.add(lq)

    remaining_physical = [p for p in range(num_q) if p not in used_physical]
    remaining_logical = [l for l in range(num_q) if l not in assigned_logical]
    for lq, pq in zip(remaining_logical, remaining_physical):
        mapping[lq] = pq
        reverse_mapping[pq] = lq

    # Step 2: Build interaction pairs list for cost computation
    interaction_pairs = []
    seen = set()
    for q1, neighbors in self.qubit_interaction_graph.items():
        for q2, weight in neighbors.items():
            if q1 < q2 and (q1, q2) not in seen:
                seen.add((q1, q2))
                interaction_pairs.append((q1, q2, weight))

    dist = self.distance_matrix

    def total_cost():
        c = 0
        for q1, q2, w in interaction_pairs:
            c += w * dist[mapping[q1]][mapping[q2]]
        return c

    # Step 3: Greedy 2-opt pair swaps until convergence
    active_logicals = sorted(assigned_logical)
    if not active_logicals:
        active_logicals = list(range(num_q))

    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_i = -1
        best_j = -1

        for idx_a in range(len(active_logicals)):
            la = active_logicals[idx_a]
            pa = mapping[la]
            for idx_b in range(idx_a + 1, len(active_logicals)):
                lb = active_logicals[idx_b]
                pb = mapping[lb]

                delta = 0
                for q1, q2, w in interaction_pairs:
                    old_p1 = mapping[q1]
                    old_p2 = mapping[q2]
                    new_p1 = old_p1
                    new_p2 = old_p2
                    if q1 == la:
                        new_p1 = pb
                    elif q1 == lb:
                        new_p1 = pa
                    if q2 == la:
                        new_p2 = pb
                    elif q2 == lb:
                        new_p2 = pa
                    delta += w * (dist[new_p1][new_p2] - dist[old_p1][old_p2])

                if delta < best_delta:
                    best_delta = delta
                    best_i = la
                    best_j = lb

        if best_delta < 0:
            pi = mapping[best_i]
            pj = mapping[best_j]
            mapping[best_i] = pj
            mapping[best_j] = pi
            reverse_mapping[pj] = best_i
            reverse_mapping[pi] = best_j
            improved = True

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    if self.use_isl:
        from src.utils.isl_data_loader import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)