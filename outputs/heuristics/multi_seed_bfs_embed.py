def init_mapping(self):
    from collections import defaultdict, deque
    import heapq

    N = self.num_qubits
    qig = self.qubit_interaction_graph

    # Collect all logical edges with weights
    seen_edges = set()
    logical_edges = []
    for q1 in qig:
        for q2, w in qig[q1].items():
            if (q1, q2) not in seen_edges:
                seen_edges.add((q1, q2))
                seen_edges.add((q2, q1))
                logical_edges.append((w, q1, q2))
    logical_edges.sort(reverse=True)

    # Score physical edges by sum of centrality
    phys_centrality = self.physical_centrality
    physical_edges = []
    for (p1, p2) in self.backend_connections:
        if p1 < p2:
            score = phys_centrality.get(p1, 0) + phys_centrality.get(p2, 0)
            physical_edges.append((score, p1, p2))
    physical_edges.sort(reverse=True)

    mapping = [-1] * N
    reverse_mapping = [-1] * N
    used_logical = set()
    used_physical = set()

    # BFS frontier: max-heap of (-weight, logical_qubit, partner_physical)
    # partner_physical is a physical qubit already placed that this logical qubit interacts with
    frontier = []
    counter = 0  # tiebreaker

    def place(lq, pq):
        mapping[lq] = pq
        reverse_mapping[pq] = lq
        used_logical.add(lq)
        used_physical.add(pq)

    def add_neighbors_to_frontier(lq):
        nonlocal counter
        for neighbor, weight in qig[lq].items():
            if neighbor not in used_logical:
                pq_partner = mapping[lq]
                heapq.heappush(frontier, (-weight, counter, neighbor, pq_partner))
                counter += 1

    # Seed top-k heaviest logical edges onto best physical edges
    phys_idx = 0
    for w, lq1, lq2 in logical_edges:
        if lq1 in used_logical or lq2 in used_logical:
            continue
        # Find next available physical edge
        while phys_idx < len(physical_edges):
            _, p1, p2 = physical_edges[phys_idx]
            phys_idx += 1
            if p1 not in used_physical and p2 not in used_physical:
                place(lq1, p1)
                place(lq2, p2)
                add_neighbors_to_frontier(lq1)
                add_neighbors_to_frontier(lq2)
                break
        else:
            break

    # Interleaved BFS: pop highest-weight frontier entry, place it
    while frontier:
        neg_w, _, lq, pq_partner = heapq.heappop(frontier)
        if lq in used_logical:
            continue

        # Find closest unused physical neighbor to pq_partner
        best_pq = None
        best_dist = float('inf')
        # BFS from pq_partner on physical graph to find nearest unused
        visited = set()
        bfs_queue = deque([pq_partner])
        visited.add(pq_partner)
        while bfs_queue:
            curr = bfs_queue.popleft()
            d = self.distance_matrix[pq_partner][curr]
            if d >= best_dist:
                continue
            if curr not in used_physical:
                best_pq = curr
                best_dist = d
                break  # BFS guarantees first found is closest
            for nb in self.backend[curr]:
                if nb not in visited:
                    visited.add(nb)
                    bfs_queue.append(nb)

        if best_pq is not None:
            place(lq, best_pq)
            add_neighbors_to_frontier(lq)

    # Fallback: assign remaining logical qubits to remaining physical qubits
    remaining_physical = [p for p in range(N) if p not in used_physical]
    remaining_logical = [q for q in range(N) if q not in used_logical]
    for lq, pq in zip(remaining_logical, remaining_physical):
        place(lq, pq)

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    if self.use_isl:
        from src.mapping.mapping import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)