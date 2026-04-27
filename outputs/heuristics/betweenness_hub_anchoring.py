def init_mapping(self):
    import networkx as nx
    from collections import defaultdict, deque

    N = self.num_qubits
    self.mapping_dict = [i for i in range(N)]
    self.reverse_mapping_dict = [i for i in range(N)]

    edge_freq = defaultdict(int)
    partners = defaultdict(lambda: defaultdict(int))
    logical_seen = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_freq[key] += 1
            partners[a][b] += 1
            partners[b][a] += 1
            logical_seen.add(a)
            logical_seen.add(b)

    if not logical_seen:
        self.mapping_dict = [i for i in range(N)]
        self.reverse_mapping_dict = [i for i in range(N)]
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    spread = {}
    for lq in logical_seen:
        s = 0
        for p, f in partners[lq].items():
            s += f
        spread[lq] = (len(partners[lq]), s)
    logical_sorted = sorted(logical_seen, key=lambda q: (spread[q][0], spread[q][1]), reverse=True)

    G = nx.Graph()
    G.add_nodes_from(range(N))
    for (u, v) in self.backend_connections:
        if u != v:
            G.add_edge(u, v)
    try:
        bc = nx.betweenness_centrality(G, normalized=True)
    except Exception:
        bc = {p: 0.0 for p in range(N)}
    deg = {p: len(self.backend[p]) if p < len(self.backend) else 0 for p in range(N)}
    phys_sorted = sorted(range(N), key=lambda p: (bc.get(p, 0.0), deg.get(p, 0)), reverse=True)

    mapping = [-1] * N
    reverse = [-1] * N
    used_phys = set()
    placed_logical = set()

    K = min(len(logical_sorted), len(phys_sorted))
    anchor_count = max(1, min(K, 2))
    for i in range(anchor_count):
        lq = logical_sorted[i]
        pq = phys_sorted[i]
        mapping[lq] = pq
        reverse[pq] = lq
        used_phys.add(pq)
        placed_logical.add(lq)

    def bfs_nearest_free(start):
        if start in used_phys:
            pass
        visited = {start}
        dq = deque([start])
        best = None
        while dq:
            node = dq.popleft()
            if node not in used_phys:
                best = node
                break
            for nb in self.backend[node]:
                if nb not in visited:
                    visited.add(nb)
                    dq.append(nb)
        if best is None:
            for p in phys_sorted:
                if p not in used_phys:
                    best = p
                    break
        return best

    for lq in logical_sorted:
        if lq in placed_logical:
            continue
        best_partner = None
        best_freq = -1
        for p, f in partners[lq].items():
            if p in placed_logical and f > best_freq:
                best_freq = f
                best_partner = p
        if best_partner is not None:
            anchor_phys = mapping[best_partner]
            target = bfs_nearest_free(anchor_phys)
        else:
            target = None
            for p in phys_sorted:
                if p not in used_phys:
                    target = p
                    break
        if target is None or target in used_phys:
            for p in range(N):
                if p not in used_phys:
                    target = p
                    break
        mapping[lq] = target
        reverse[target] = lq
        used_phys.add(target)
        placed_logical.add(lq)

    free_phys = [p for p in range(N) if p not in used_phys]
    fi = 0
    for lq in range(N):
        if mapping[lq] == -1:
            if fi < len(free_phys):
                pq = free_phys[fi]
                fi += 1
            else:
                for p in range(N):
                    if p not in used_phys:
                        pq = p
                        break
            mapping[lq] = pq
            reverse[pq] = lq
            used_phys.add(pq)

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)