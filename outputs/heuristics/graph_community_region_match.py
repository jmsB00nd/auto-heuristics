def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    num_q = self.num_qubits

    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    logical_qubits = set()
    for q1, neighbors in self.qubit_interaction_graph.items():
        logical_qubits.add(q1)
        for q2 in neighbors:
            logical_qubits.add(q2)

    if not logical_qubits:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    G_logical = nx.Graph()
    for q1, neighbors in self.qubit_interaction_graph.items():
        for q2, w in neighbors.items():
            if q1 < q2:
                G_logical.add_edge(q1, q2, weight=w)
    for q in logical_qubits:
        if q not in G_logical:
            G_logical.add_node(q)

    G_physical = nx.Graph()
    for node, neighbors in self.backend.items():
        for nb in neighbors:
            if node < nb:
                G_physical.add_edge(node, nb)

    try:
        logical_communities_map = nx.community.louvain_communities(G_logical, weight='weight', seed=42)
    except Exception:
        logical_communities_map = [set(G_logical.nodes())]
    logical_communities = [list(c) for c in logical_communities_map]

    try:
        physical_regions_map = nx.community.louvain_communities(G_physical, seed=42)
    except Exception:
        physical_regions_map = [set(G_physical.nodes())]
    physical_regions = [list(r) for r in physical_regions_map]

    def logical_community_score(comm):
        total_weight = 0
        for q in comm:
            if q in self.qubit_interaction_graph:
                total_weight += sum(self.qubit_interaction_graph[q].values())
        return (len(comm), total_weight)

    def physical_region_score(region):
        region_set = set(region)
        internal_edges = 0
        for q in region:
            if q in self.backend:
                internal_edges += len(self.backend[q] & region_set)
        return (len(region), internal_edges)

    logical_communities.sort(key=logical_community_score, reverse=True)
    physical_regions.sort(key=physical_region_score, reverse=True)

    used_physical = set()
    assigned_logical = set()

    for i, comm in enumerate(logical_communities):
        if i < len(physical_regions):
            region = physical_regions[i]
        else:
            region = [p for p in range(num_q) if p not in used_physical]

        def logical_degree(q):
            if q in self.qubit_interaction_graph:
                return sum(self.qubit_interaction_graph[q].values())
            return 0

        region_set = set(region)
        def physical_degree(p):
            if p in self.backend:
                return len(self.backend[p] & region_set)
            return 0

        sorted_logical = sorted(comm, key=logical_degree, reverse=True)
        available_in_region = [p for p in region if p not in used_physical]
        available_in_region.sort(key=physical_degree, reverse=True)

        for j, lq in enumerate(sorted_logical):
            if j < len(available_in_region):
                pq = available_in_region[j]
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_physical.add(pq)
                assigned_logical.add(lq)

    remaining_logical = [q for q in range(num_q) if q not in assigned_logical]
    remaining_physical = [p for p in range(num_q) if p not in used_physical]
    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)