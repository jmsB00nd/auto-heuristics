def init_mapping(self):
    num_q = self.num_qubits
    num_physical = len(self.distance_matrix)

    pairs = {}
    for q1, neighbors in self.temporal_interaction_graph.items():
        for q2, weight in neighbors.items():
            key = (min(q1, q2), max(q1, q2))
            if key not in pairs:
                pairs[key] = weight

    sorted_pairs = sorted(pairs.items(), key=lambda x: -x[1])

    logical_to_physical = {}
    physical_used = set()
    all_physical = set(range(num_physical))

    for (q1, q2), _ in sorted_pairs:
        q1_placed = q1 in logical_to_physical
        q2_placed = q2 in logical_to_physical

        if q1_placed and q2_placed:
            continue

        if not q1_placed and not q2_placed:
            best_edge = None
            best_score = -1.0
            for (p1, p2) in self.backend_connections:
                if p1 not in physical_used and p2 not in physical_used:
                    score = self.physical_centrality.get(p1, 0) + self.physical_centrality.get(p2, 0)
                    if score > best_score:
                        best_score = score
                        best_edge = (p1, p2)
            if best_edge is not None:
                logical_to_physical[q1] = best_edge[0]
                logical_to_physical[q2] = best_edge[1]
                physical_used.add(best_edge[0])
                physical_used.add(best_edge[1])
            else:
                available = sorted(all_physical - physical_used)
                if available:
                    p = available[0]
                    logical_to_physical[q1] = p
                    physical_used.add(p)
                available = sorted(all_physical - physical_used)
                if available:
                    p = available[0]
                    logical_to_physical[q2] = p
                    physical_used.add(p)
        else:
            placed_q, unplaced_q = (q1, q2) if q1_placed else (q2, q1)
            anchor = logical_to_physical[placed_q]
            available = all_physical - physical_used
            if available:
                p = min(available, key=lambda x: self.distance_matrix[anchor][x])
                logical_to_physical[unplaced_q] = p
                physical_used.add(p)

    self.mapping_dict = [None] * num_q
    self.reverse_mapping_dict = [None] * num_q

    for lq, pq in logical_to_physical.items():
        if lq < num_q and pq < num_q:
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq

    remaining_logical = [i for i in range(num_q) if self.mapping_dict[i] is None]
    remaining_physical = [i for i in range(num_q) if self.reverse_mapping_dict[i] is None]

    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)