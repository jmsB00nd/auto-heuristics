def init_mapping(self):
    import heapq
    from collections import defaultdict

    N = self.num_qubits

    logical_set = set()
    edge_weight = defaultdict(int)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_set.add(a)
            logical_set.add(b)
            key = (a, b) if a < b else (b, a)
            edge_weight[key] += 1

    if hasattr(self, "qubit_interaction_graph") and self.qubit_interaction_graph:
        for u, nbrs in self.qubit_interaction_graph.items():
            for v, w in nbrs.items():
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                if w > edge_weight[key]:
                    edge_weight[key] = w
                logical_set.add(u)
                logical_set.add(v)

    self.mapping_dict = [i for i in range(N)]
    self.reverse_mapping_dict = [i for i in range(N)]

    if not edge_weight or not logical_set:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    sorted_edges = sorted(edge_weight.items(), key=lambda x: -x[1])
    cap = max(8, min(len(sorted_edges), 2 * N))
    top_edges = sorted_edges[:cap]

    L_adj = defaultdict(set)
    for (u, v), _ in top_edges:
        L_adj[u].add(v)
        L_adj[v].add(u)
    L_nodes = sorted(L_adj.keys(), key=lambda q: -len(L_adj[q]))

    H_adj = defaultdict(set)
    for p, nbrs in self.backend.items():
        for q in nbrs:
            if p != q:
                H_adj[p].add(q)
    H_nodes = list(range(N))

    centrality = getattr(self, "physical_centrality", {}) or {}

    best = {"map": {}, "size": 0}

    def degree_in_remaining(node, adj, allowed):
        return sum(1 for x in adj[node] if x in allowed)

    def edges_in_partial(partial_L, adj):
        s = 0
        nodes = set(partial_L)
        for u in partial_L:
            for v in adj[u]:
                if v in nodes:
                    s += 1
        return s // 2

    max_depth = min(len(L_nodes), N, 24)
    node_limit = [4000]

    def branch(partial_map, used_L, used_P, candidates):
        if node_limit[0] <= 0:
            return
        node_limit[0] -= 1

        common_edges = 0
        for u in partial_map:
            pu = partial_map[u]
            for v in L_adj[u]:
                if v in partial_map and v > u:
                    if partial_map[v] in H_adj[pu]:
                        common_edges += 1

        cur_size = len(partial_map) + common_edges * 0
        score = common_edges
        if score > best["size"]:
            best["size"] = score
            best["map"] = dict(partial_map)

        if len(partial_map) >= max_depth:
            return
        if not candidates:
            return

        next_L = None
        for cand in candidates:
            if cand[0] not in used_L:
                next_L = cand[0]
                break
        if next_L is None:
            return

        l_node = next_L
        l_neighbors_mapped = [partial_map[n] for n in L_adj[l_node] if n in partial_map]

        phys_candidates = []
        for p in H_nodes:
            if p in used_P:
                continue
            ok = True
            for pn in l_neighbors_mapped:
                if pn not in H_adj[p]:
                    ok = False
                    break
            if ok:
                phys_candidates.append(p)

        phys_candidates.sort(key=lambda p: (-len(H_adj[p] & set(range(N))), -centrality.get(p, 0.0)))
        phys_candidates = phys_candidates[:8]

        for p in phys_candidates:
            partial_map[l_node] = p
            used_L.add(l_node)
            used_P.add(p)
            new_candidates = [c for c in candidates if c[0] != l_node]
            branch(partial_map, used_L, used_P, new_candidates)
            del partial_map[l_node]
            used_L.discard(l_node)
            used_P.discard(p)

        new_candidates = [c for c in candidates if c[0] != l_node]
        branch(partial_map, used_L, used_P, new_candidates)

    init_candidates = [(l, len(L_adj[l])) for l in L_nodes]
    init_candidates.sort(key=lambda x: -x[1])
    branch({}, set(), set(), init_candidates)

    seed_map = best["map"]

    used_L = set(seed_map.keys())
    used_P = set(seed_map.values())

    remaining_logicals = [l for l in logical_set if l not in used_L]
    activity = getattr(self, "logical_activity", {}) or {}
    remaining_logicals.sort(key=lambda q: -activity.get(q, 0))

    final_map = dict(seed_map)

    for l in remaining_logicals:
        mapped_neighbors = []
        for nb in L_adj.get(l, ()):
            if nb in final_map:
                mapped_neighbors.append(final_map[nb])

        best_p = None
        best_score = None
        for p in H_nodes:
            if p in used_P:
                continue
            adj_score = 0
            dist_pen = 0.0
            for pn in mapped_neighbors:
                if pn in H_adj[p]:
                    adj_score += 1
                else:
                    d = self.distance_matrix[p][pn] if pn < N and p < N else 1.0
                    dist_pen += d
            cent = centrality.get(p, 0.0)
            score = (adj_score, cent, -dist_pen)
            if best_score is None or score > best_score:
                best_score = score
                best_p = p

        if best_p is None:
            for p in H_nodes:
                if p not in used_P:
                    best_p = p
                    break
        if best_p is None:
            continue
        final_map[l] = best_p
        used_P.add(best_p)
        used_L.add(l)

    mapping = [None] * N
    for l, p in final_map.items():
        if 0 <= l < N and 0 <= p < N:
            mapping[l] = p

    used_phys = set(p for p in mapping if p is not None)
    free_phys = [p for p in range(N) if p not in used_phys]
    fp_iter = iter(free_phys)
    for i in range(N):
        if mapping[i] is None:
            try:
                mapping[i] = next(fp_iter)
            except StopIteration:
                for p in range(N):
                    if p not in used_phys:
                        mapping[i] = p
                        used_phys.add(p)
                        break

    seen = set()
    duplicates = []
    for i, p in enumerate(mapping):
        if p in seen:
            duplicates.append(i)
        else:
            seen.add(p)
    if duplicates:
        free_phys = [p for p in range(N) if p not in seen]
        for i in duplicates:
            if free_phys:
                np = free_phys.pop()
                mapping[i] = np
                seen.add(np)

    self.mapping_dict = mapping
    self.reverse_mapping_dict = [0] * N
    for l, p in enumerate(mapping):
        self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)