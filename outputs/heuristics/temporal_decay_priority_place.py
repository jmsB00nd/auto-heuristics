def init_mapping(self):
    N = self.num_qubits

    pair_weights = {}
    for q1, neighbors in self.temporal_interaction_graph.items():
        for q2, w in neighbors.items():
            if q1 < q2:
                pair_weights[(q1, q2)] = pair_weights.get((q1, q2), 0.0) + w
            elif q2 < q1:
                pair_weights[(q2, q1)] = pair_weights.get((q2, q1), 0.0) + w

    sorted_pairs = sorted(pair_weights.items(), key=lambda kv: -kv[1])

    physical_edges = []
    num_phys = len(self.distance_matrix)
    for u in range(num_phys):
        for v in range(u + 1, num_phys):
            if self.distance_matrix[u][v] == 1:
                cu = self.physical_centrality.get(u, 0.0)
                cv = self.physical_centrality.get(v, 0.0)
                physical_edges.append((-(cu + cv), u, v))
    physical_edges.sort()
    edge_iter_idx = [0]

    mapping = [-1] * N
    reverse = [-1] * N
    used_physical = set()
    placed_logical = set()

    def next_free_edge():
        while edge_iter_idx[0] < len(physical_edges):
            _, u, v = physical_edges[edge_iter_idx[0]]
            edge_iter_idx[0] += 1
            if u not in used_physical and v not in used_physical:
                return (u, v)
        return None

    def place(lq, pq):
        mapping[lq] = pq
        reverse[pq] = lq
        used_physical.add(pq)
        placed_logical.add(lq)

    def closest_unused_to(pq_anchor):
        best_p = None
        best_d = None
        for p in range(num_phys):
            if p in used_physical:
                continue
            d = self.distance_matrix[pq_anchor][p]
            if best_d is None or d < best_d:
                best_d = d
                best_p = p
        return best_p

    for (lq1, lq2), _w in sorted_pairs:
        if lq1 >= N or lq2 >= N:
            continue
        l1_done = lq1 in placed_logical
        l2_done = lq2 in placed_logical
        if l1_done and l2_done:
            continue
        if not l1_done and not l2_done:
            edge = next_free_edge()
            if edge is None:
                anchor = closest_unused_to(0)
                if anchor is None:
                    break
                place(lq1, anchor)
                partner = closest_unused_to(anchor)
                if partner is None:
                    break
                place(lq2, partner)
            else:
                u, v = edge
                cu = self.physical_centrality.get(u, 0.0)
                cv = self.physical_centrality.get(v, 0.0)
                if cu >= cv:
                    place(lq1, u)
                    place(lq2, v)
                else:
                    place(lq1, v)
                    place(lq2, u)
        else:
            anchored_l = lq1 if l1_done else lq2
            free_l = lq2 if l1_done else lq1
            anchor_p = mapping[anchored_l]
            target = closest_unused_to(anchor_p)
            if target is None:
                continue
            place(free_l, target)

    logical_in_circuit = set()
    for gate_qubits in self.access.values():
        for q in gate_qubits:
            if 0 <= q < N:
                logical_in_circuit.add(q)

    remaining_logical = [q for q in range(N) if q not in placed_logical and q in logical_in_circuit]
    remaining_logical += [q for q in range(N) if q not in placed_logical and q not in logical_in_circuit]

    remaining_physical = [p for p in range(N) if p not in used_physical]
    remaining_physical.sort(key=lambda p: -self.physical_centrality.get(p, 0.0))

    for lq, pq in zip(remaining_logical, remaining_physical):
        place(lq, pq)

    leftover_logical = [q for q in range(N) if q not in placed_logical]
    leftover_physical = [p for p in range(N) if p not in used_physical]
    for lq, pq in zip(leftover_logical, leftover_physical):
        place(lq, pq)

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    if self.use_isl:
        try:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        except Exception:
            self.isl_mapping = None

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)