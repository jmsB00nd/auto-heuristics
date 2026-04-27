def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    N = self.num_qubits

    edge_weight = defaultdict(float)
    node_weight = defaultdict(float)
    logical_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a == b:
                logical_nodes.add(a)
                continue
            u, v = (a, b) if a < b else (b, a)
            edge_weight[(u, v)] += 1.0
            node_weight[a] += 1.0
            node_weight[b] += 1.0
            logical_nodes.add(a)
            logical_nodes.add(b)
        elif len(qubits) == 1:
            logical_nodes.add(int(qubits[0]))

    G = nx.Graph()
    G.add_nodes_from(logical_nodes)
    for (u, v), w in edge_weight.items():
        G.add_edge(u, v, weight=w)

    try:
        matching = nx.max_weight_matching(G, maxcardinality=False)
    except Exception:
        matching = set()

    matched_pairs = []
    matched_logical = set()
    for (u, v) in matching:
        w = edge_weight.get((u, v) if u < v else (v, u), 0.0)
        matched_pairs.append((w, u, v))
        matched_logical.add(u)
        matched_logical.add(v)
    matched_pairs.sort(key=lambda t: -t[0])

    phys_edges = set()
    for (p, q) in self.backend_connections:
        if p == q:
            continue
        a, b = (p, q) if p < q else (q, p)
        phys_edges.add((a, b))

    def deg(p):
        try:
            return len(self.backend[p])
        except Exception:
            return 0

    edge_score = []
    for (a, b) in phys_edges:
        edge_score.append((deg(a) + deg(b), a, b))
    edge_score.sort(key=lambda t: -t[0])

    mapping = [-1] * N
    reverse = [-1] * N
    used_physical = set()
    placed_logical = set()

    def place(L, P):
        mapping[L] = P
        reverse[P] = L
        used_physical.add(P)
        placed_logical.add(L)

    available_edges = list(edge_score)
    for (w, lu, lv) in matched_pairs:
        chosen_idx = -1
        best_score = None
        for idx, (sc, pa, pb) in enumerate(available_edges):
            if pa in used_physical or pb in used_physical:
                continue
            score = sc
            if best_score is None or score > best_score:
                best_score = score
                chosen_idx = idx
                break
        if chosen_idx == -1:
            continue
        sc, pa, pb = available_edges[chosen_idx]
        nbr_pa = sum(1 for n in self.backend[pa] if n in used_physical)
        nbr_pb = sum(1 for n in self.backend[pb] if n in used_physical)
        if nbr_pb > nbr_pa:
            place(lu, pb)
            place(lv, pa)
        else:
            place(lu, pa)
            place(lv, pb)

    singletons = [L for L in logical_nodes if L not in placed_logical and L < N]
    singletons.sort(key=lambda L: -node_weight.get(L, 0.0))

    for L in singletons:
        best_p = -1
        best_adj = -1
        for P in range(N):
            if P in used_physical:
                continue
            adj = sum(1 for nb in self.backend[P] if nb in used_physical)
            if adj > best_adj:
                best_adj = adj
                best_p = P
        if best_p == -1:
            for P in range(N):
                if P not in used_physical:
                    best_p = P
                    break
        if best_p != -1:
            place(L, best_p)

    for L in range(N):
        if mapping[L] == -1:
            if L not in used_physical:
                mapping[L] = L
                reverse[L] = L
                used_physical.add(L)
            else:
                for P in range(N):
                    if P not in used_physical:
                        mapping[L] = P
                        reverse[P] = L
                        used_physical.add(P)
                        break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)