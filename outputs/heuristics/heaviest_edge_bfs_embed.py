def init_mapping(self):
    import networkx as nx
    from collections import deque

    num_q = self.num_qubits

    # Build networkx graph for the physical backend
    G_phys = nx.Graph()
    for (u, v) in self.backend_connections:
        G_phys.add_edge(u, v)

    # Compute betweenness centrality for physical qubits
    phys_centrality = nx.betweenness_centrality(G_phys)

    # Find the heaviest edge in the logical interaction graph
    best_weight = -1
    best_logical_edge = None
    seen = set()
    for q1, neighbors in self.qubit_interaction_graph.items():
        for q2, w in neighbors.items():
            if (q2, q1) not in seen:
                seen.add((q1, q2))
                if w > best_weight:
                    best_weight = w
                    best_logical_edge = (q1, q2)

    # Find the highest-centrality physical edge
    best_phys_score = -1
    best_phys_edge = None
    for (u, v) in self.backend_connections:
        score = phys_centrality.get(u, 0) + phys_centrality.get(v, 0)
        if score > best_phys_score:
            best_phys_score = score
            best_phys_edge = (u, v)

    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    if best_logical_edge is None or best_phys_edge is None:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    mapped_logical = set()
    used_physical = set()

    l1, l2 = best_logical_edge
    p1, p2 = best_phys_edge

    self.mapping_dict[l1] = p1
    self.mapping_dict[l2] = p2
    self.reverse_mapping_dict[p1] = l1
    self.reverse_mapping_dict[p2] = l2
    mapped_logical.add(l1)
    mapped_logical.add(l2)
    used_physical.add(p1)
    used_physical.add(p2)

    # BFS on the interaction graph, prioritized by interaction weight
    bfs_queue = deque()
    # Seed BFS from both placed qubits
    for seed in [l1, l2]:
        for neighbor_l, w in sorted(self.qubit_interaction_graph[seed].items(),
                                     key=lambda x: -x[1]):
            if neighbor_l not in mapped_logical:
                bfs_queue.append((neighbor_l, seed))

    visited_in_queue = set(mapped_logical)
    for (nl, _) in bfs_queue:
        visited_in_queue.add(nl)

    while bfs_queue:
        logical_q, partner_l = bfs_queue.popleft()
        if logical_q in mapped_logical:
            continue

        # Find closest available physical neighbor to partner's physical qubit
        partner_p = self.mapping_dict[partner_l]
        best_p = None
        best_dist = float('inf')

        # BFS on physical graph from partner_p to find nearest unused
        phys_bfs = deque([partner_p])
        phys_visited = {partner_p}
        while phys_bfs:
            curr_p = phys_bfs.popleft()
            if curr_p not in used_physical:
                best_p = curr_p
                break
            for adj_p in self.backend.get(curr_p, []):
                if adj_p not in phys_visited:
                    phys_visited.add(adj_p)
                    phys_bfs.append(adj_p)

        if best_p is None:
            continue

        self.mapping_dict[logical_q] = best_p
        self.reverse_mapping_dict[best_p] = logical_q
        mapped_logical.add(logical_q)
        used_physical.add(best_p)

        # Enqueue neighbors of this newly placed logical qubit
        for neighbor_l, w in sorted(self.qubit_interaction_graph[logical_q].items(),
                                     key=lambda x: -x[1]):
            if neighbor_l not in visited_in_queue:
                visited_in_queue.add(neighbor_l)
                bfs_queue.append((neighbor_l, logical_q))

    # Fallback: assign any unmapped logical qubits to remaining physical qubits
    remaining_physical = [p for p in range(num_q) if p not in used_physical]
    remaining_logical = [l for l in range(num_q) if l not in mapped_logical]
    for l, p in zip(remaining_logical, remaining_physical):
        self.mapping_dict[l] = p
        self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)