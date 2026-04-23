def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    n = self.num_qubits

    # Build logical interaction graph (networkx)
    LG = nx.Graph()
    for q1 in self.qubit_interaction_graph:
        for q2, w in self.qubit_interaction_graph[q1].items():
            if q1 < q2:
                LG.add_edge(q1, q2, weight=w)

    logical_qubits_with_interactions = set(LG.nodes())

    # --- Extract multiple spines greedily ---
    spines = []
    remaining = LG.copy()
    while remaining.number_of_edges() > 0:
        # Find heaviest edge to seed the spine
        best_edge = max(remaining.edges(data=True), key=lambda e: e[2]['weight'])
        spine = [best_edge[0], best_edge[1]]
        spine_weight = best_edge[2]['weight']
        spine_nodes = set(spine)

        # Extend spine in both directions greedily
        for direction in [0, -1]:  # 0 = prepend, -1 = append
            while True:
                tip = spine[direction]
                best_next = None
                best_w = 0
                for nb in remaining.neighbors(tip):
                    if nb not in spine_nodes:
                        w = remaining[tip][nb]['weight']
                        if w > best_w:
                            best_w = w
                            best_next = nb
                if best_next is None:
                    break
                spine_weight += best_w
                spine_nodes.add(best_next)
                if direction == 0:
                    spine.insert(0, best_next)
                else:
                    spine.append(best_next)

        spines.append((spine_weight, spine))
        remaining.remove_nodes_from(spine_nodes)

    # Sort spines by total weight descending
    spines.sort(key=lambda x: -x[0])

    # --- Build physical coupling graph ---
    PG = nx.Graph()
    for u, neighbors in self.backend.items():
        for v in neighbors:
            PG.add_edge(u, v)

    # Physical qubits sorted by centrality descending
    phys_by_centrality = sorted(self.physical_centrality.keys(),
                                key=lambda p: -self.physical_centrality[p])

    used_physical = set()
    mapping = [None] * n
    reverse = [None] * n

    def assign(logical_q, physical_q):
        mapping[logical_q] = physical_q
        reverse[physical_q] = logical_q
        used_physical.add(physical_q)

    # --- Embed each spine onto a disjoint physical path ---
    for _, spine in spines:
        spine_len = len(spine)
        best_path = None

        # Try starting from highest-centrality unused physical qubits
        for start_p in phys_by_centrality:
            if start_p in used_physical:
                continue
            # BFS/DFS to find a path of length spine_len using only unused physical qubits
            path = [start_p]
            visited = {start_p}
            stack = [(start_p, list(PG.neighbors(start_p)))]
            while stack and len(path) < spine_len:
                node, neighbors = stack[-1]
                found = False
                while neighbors:
                    nb = neighbors.pop()
                    if nb not in used_physical and nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        if len(path) == spine_len:
                            found = True
                            break
                        stack.append((nb, list(PG.neighbors(nb))))
                        found = True
                        break
                if not found:
                    if len(path) > 1:
                        path.pop()
                    if stack:
                        stack.pop()
                    else:
                        break
            if len(path) == spine_len:
                best_path = path
                break

        if best_path is not None and len(best_path) == spine_len:
            for i, lq in enumerate(spine):
                assign(lq, best_path[i])
        else:
            # Could not find a contiguous path; place individually
            for lq in spine:
                if mapping[lq] is not None:
                    continue
                best_p = None
                best_score = float('inf')
                for p in phys_by_centrality:
                    if p in used_physical:
                        continue
                    score = -self.physical_centrality[p]
                    if score < best_score:
                        best_score = score
                        best_p = p
                if best_p is not None:
                    assign(lq, best_p)

    # --- Place residual logical qubits (those with interactions but not in any spine) ---
    residual = [q for q in logical_qubits_with_interactions if mapping[q] is None]
    # Sort residual by activity descending
    residual.sort(key=lambda q: -self.logical_activity.get(q, 0))

    for lq in residual:
        # Find heaviest placed neighbor
        best_w = 0
        heaviest_neighbor_phys = None
        for nb, w in self.qubit_interaction_graph[lq].items():
            if mapping[nb] is not None and w > best_w:
                best_w = w
                heaviest_neighbor_phys = mapping[nb]

        best_p = None
        best_dist = float('inf')
        for p in range(n):
            if p in used_physical or p not in self.physical_centrality:
                continue
            if heaviest_neighbor_phys is not None:
                d = self.distance_matrix[p][heaviest_neighbor_phys]
            else:
                d = -self.physical_centrality.get(p, 0)
            if d < best_dist:
                best_dist = d
                best_p = p
        if best_p is not None:
            assign(lq, best_p)

    # --- Identity fallback for any remaining qubits ---
    unassigned_logical = [q for q in range(n) if mapping[q] is None]
    unused_phys = sorted(p for p in range(n) if p not in used_physical)
    for i, lq in enumerate(unassigned_logical):
        assign(lq, unused_phys[i])

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)