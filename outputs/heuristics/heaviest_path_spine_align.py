def init_mapping(self):
    import networkx as nx

    n = self.num_qubits

    G_logical = nx.Graph()
    for q1 in self.qubit_interaction_graph:
        for q2, w in self.qubit_interaction_graph[q1].items():
            if q1 < q2:
                G_logical.add_edge(q1, q2, weight=w)

    if G_logical.number_of_edges() == 0:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    start = max(G_logical.nodes(),
                key=lambda v: sum(d['weight'] for _, _, d in G_logical.edges(v, data=True)))

    spine = [start]
    visited = {start}

    while True:
        cur = spine[-1]
        cands = [(v, G_logical[cur][v]['weight'])
                 for v in G_logical.neighbors(cur) if v not in visited]
        if not cands:
            break
        best_v, _ = max(cands, key=lambda x: x[1])
        spine.append(best_v)
        visited.add(best_v)

    while True:
        cur = spine[0]
        cands = [(v, G_logical[cur][v]['weight'])
                 for v in G_logical.neighbors(cur) if v not in visited]
        if not cands:
            break
        best_v, _ = max(cands, key=lambda x: x[1])
        spine.insert(0, best_v)
        visited.add(best_v)

    G_physical = nx.Graph()
    for (u, v) in self.backend_connections:
        G_physical.add_edge(u, v)

    spine_len = len(spine)
    physical_nodes = sorted(G_physical.nodes(),
                            key=lambda v: self.physical_centrality.get(v, 0),
                            reverse=True)

    best_path = None
    best_score = -1

    for start_phys in physical_nodes[:min(len(physical_nodes), 20)]:
        path = [start_phys]
        used = {start_phys}

        while len(path) < spine_len:
            cur = path[-1]
            nbrs = [v for v in G_physical.neighbors(cur) if v not in used]
            if not nbrs:
                break
            nxt = max(nbrs, key=lambda v: G_physical.degree(v))
            path.append(nxt)
            used.add(nxt)

        while len(path) < spine_len:
            cur = path[0]
            nbrs = [v for v in G_physical.neighbors(cur) if v not in used]
            if not nbrs:
                break
            nxt = max(nbrs, key=lambda v: G_physical.degree(v))
            path.insert(0, nxt)
            used.add(nxt)

        if len(path) < spine_len:
            continue

        score = sum(self.physical_centrality.get(p, 0) for p in path)
        if score > best_score:
            best_score = score
            best_path = path

    mapping = [None] * n
    reverse_mapping = [None] * n
    used_physical = set()

    if best_path and len(best_path) >= spine_len:
        for i, lq in enumerate(spine):
            pq = best_path[i]
            mapping[lq] = pq
            reverse_mapping[pq] = lq
            used_physical.add(pq)
    else:
        avail = sorted(set(range(n)) - used_physical)
        for i, lq in enumerate(spine):
            if i < len(avail):
                pq = avail[i]
                mapping[lq] = pq
                reverse_mapping[pq] = lq
                used_physical.add(pq)

    unmapped_logical = [q for q in range(n) if mapping[q] is None]
    available_physical = set(q for q in range(n) if reverse_mapping[q] is None)

    qig = self.qubit_interaction_graph
    unmapped_logical.sort(
        key=lambda lq: sum(qig.get(lq, {}).get(sq, 0) for sq in spine),
        reverse=True)

    dm = self.distance_matrix
    dm_size = len(dm)

    for lq in unmapped_logical:
        best_phys = None
        best_cost = float('inf')

        for pq in available_physical:
            cost = 0
            for sq in spine:
                w = qig.get(lq, {}).get(sq, 0)
                if w > 0 and pq < dm_size and mapping[sq] < dm_size:
                    cost += w * dm[pq][mapping[sq]]
            if cost < best_cost:
                best_cost = cost
                best_phys = pq

        if best_phys is None:
            best_phys = min(available_physical)

        mapping[lq] = best_phys
        reverse_mapping[best_phys] = lq
        available_physical.discard(best_phys)

    for q in range(n):
        if mapping[q] is None:
            for p in range(n):
                if reverse_mapping[p] is None:
                    mapping[q] = p
                    reverse_mapping[p] = q
                    break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)