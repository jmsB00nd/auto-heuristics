def init_mapping(self):
    import heapq

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    logical_qubits = set()
    for _gid, qubits in self.access.items():
        for q in qubits:
            if 0 <= q < N:
                logical_qubits.add(q)

    if not logical_qubits:
        for L in range(N):
            self.mapping_dict[L] = L
            self.reverse_mapping_dict[L] = L
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    activity = {L: self.logical_activity.get(L, 0) for L in logical_qubits}
    sorted_logicals = sorted(logical_qubits, key=lambda L: (-activity.get(L, 0), L))

    opening_cost = {}
    for p in range(N):
        c = self.physical_centrality.get(p, 0.0)
        opening_cost[p] = (1.0 / c) if c and c > 0 else 1e12

    used_physicals = set()

    for L in sorted_logicals:
        neighbors = self.qubit_interaction_graph.get(L, {})
        placed_neighbors = [
            (v, w) for v, w in neighbors.items()
            if 0 <= v < N and self.mapping_dict[v] != -1 and w > 0
        ]

        best_p = -1

        if not placed_neighbors:
            best_open = float('inf')
            for p in range(N):
                if p in used_physicals:
                    continue
                oc = opening_cost[p]
                if oc < best_open:
                    best_open = oc
                    best_p = p
        else:
            heap = []
            for p in range(N):
                if p in used_physicals:
                    continue
                heapq.heappush(heap, (opening_cost[p], p))

            while heap:
                est, p = heapq.heappop(heap)
                if p in used_physicals:
                    continue
                conn = 0.0
                for v, w in placed_neighbors:
                    conn += w * self.distance_matrix[p][self.mapping_dict[v]]
                exact = opening_cost[p] + conn

                if not heap or exact <= heap[0][0]:
                    best_p = p
                    break
                heapq.heappush(heap, (exact, p))

        if best_p == -1:
            for p in range(N):
                if p not in used_physicals:
                    best_p = p
                    break

        if best_p != -1:
            self.mapping_dict[L] = best_p
            self.reverse_mapping_dict[best_p] = L
            used_physicals.add(best_p)

    unused = [p for p in range(N) if p not in used_physicals]
    ui = 0
    for L in range(N):
        if self.mapping_dict[L] == -1 and ui < len(unused):
            p = unused[ui]
            self.mapping_dict[L] = p
            self.reverse_mapping_dict[p] = L
            ui += 1

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)