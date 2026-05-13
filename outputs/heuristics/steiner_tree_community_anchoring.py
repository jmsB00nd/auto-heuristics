def init_mapping(self):
    import networkx as nx
    from networkx.algorithms.approximation import steiner_tree
    try:
        from networkx.algorithms.community import louvain_communities
    except Exception:
        louvain_communities = None

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # Collect logical qubits used by the circuit
    logical_set = set()
    for gate_id, qubits in self.access.items():
        for q in qubits:
            if 0 <= q < N:
                logical_set.add(q)

    # Build weighted logical interaction graph
    L = nx.Graph()
    L.add_nodes_from(logical_set)
    for q1 in self.qubit_interaction_graph:
        if q1 >= N:
            continue
        for q2, w in self.qubit_interaction_graph[q1].items():
            if q2 >= N or q1 >= q2 or w <= 0:
                continue
            if q1 in logical_set or q2 in logical_set:
                L.add_node(q1); L.add_node(q2)
                L.add_edge(q1, q2, weight=float(w))

    # Detect communities
    communities = []
    if louvain_communities is not None and L.number_of_nodes() > 0 and L.number_of_edges() > 0:
        try:
            communities = [set(c) for c in louvain_communities(L, weight='weight', seed=0)]
        except Exception:
            communities = []
    if not communities:
        communities = [{q} for q in logical_set] if logical_set else []

    communities = sorted(communities, key=lambda c: -len(c))

    # Hardware graph
    H = nx.Graph()
    H.add_nodes_from(range(N))
    for u, nbrs in self.backend.items():
        for v in nbrs:
            if u != v:
                H.add_edge(int(u), int(v))

    used_phys = set()
    used_log = set()

    phys_by_centrality = sorted(range(N), key=lambda p: -self.physical_centrality.get(p, 0.0))

    def internal_weight(q, comm_set):
        total = 0.0
        for q2, w in self.qubit_interaction_graph[q].items():
            if q2 in comm_set:
                total += w
        return total

    for community in communities:
        community = [q for q in community if 0 <= q < N and q not in used_log]
        if not community:
            continue
        k = len(community)

        # Anchor set: most central unused
        num_anchors = max(1, min(k, 3))
        anchors = []
        for p in phys_by_centrality:
            if p not in used_phys:
                anchors.append(p)
                if len(anchors) >= num_anchors:
                    break
        if not anchors:
            continue

        # Steiner tree on hardware with usage-penalized weights
        tree_nodes = list(anchors)
        if len(anchors) >= 2 and H.number_of_edges() > 0:
            try:
                H_w = nx.Graph()
                H_w.add_nodes_from(H.nodes())
                for u, v in H.edges():
                    w = 1.0
                    if u in used_phys: w += 10.0
                    if v in used_phys: w += 10.0
                    H_w.add_edge(u, v, weight=w)
                tree = steiner_tree(H_w, anchors, weight='weight')
                tree_nodes = list(tree.nodes())
            except Exception:
                tree_nodes = list(anchors)

        avail = [p for p in tree_nodes if p not in used_phys]

        # BFS-expand if too few unused tree nodes
        if len(avail) < k:
            visited = set(avail) | used_phys | set(tree_nodes)
            frontier = list(tree_nodes)
            while len(avail) < k and frontier:
                nxt = []
                for u in frontier:
                    for v in self.backend.get(u, ()):
                        if v in visited:
                            continue
                        visited.add(v)
                        nxt.append(v)
                        if v not in used_phys:
                            avail.append(v)
                            if len(avail) >= k:
                                break
                    if len(avail) >= k:
                        break
                frontier = nxt

        # Last-resort top-up by centrality
        if len(avail) < k:
            seen = set(avail)
            for p in phys_by_centrality:
                if p not in used_phys and p not in seen:
                    avail.append(p); seen.add(p)
                    if len(avail) >= k:
                        break

        # Rank logical members by intra-community weight
        comm_set = set(community)
        sorted_comm = sorted(community, key=lambda q: -internal_weight(q, comm_set))

        # Rank physicals: anchors first, then other tree nodes, then expansion; centrality breaks ties
        anchor_set = set(anchors)
        tree_set = set(tree_nodes)
        def avail_score(p):
            tier = 0 if p in anchor_set else (1 if p in tree_set else 2)
            return (tier, -self.physical_centrality.get(p, 0.0))
        sorted_avail = sorted(avail, key=avail_score)

        for q, p in zip(sorted_comm, sorted_avail):
            if q in used_log or p in used_phys:
                continue
            self.mapping_dict[q] = p
            self.reverse_mapping_dict[p] = q
            used_log.add(q)
            used_phys.add(p)

    # Fallback: assign any logical still unmapped to a free physical (identity preferred)
    unused_phys = [p for p in range(N) if p not in used_phys]
    unused_set = set(unused_phys)
    for q in range(N):
        if self.mapping_dict[q] != -1:
            continue
        if q in unused_set:
            p = q
            unused_set.remove(p)
            unused_phys.remove(p)
        else:
            p = unused_phys.pop(0)
            unused_set.discard(p)
        self.mapping_dict[q] = p
        self.reverse_mapping_dict[p] = q
        used_phys.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)