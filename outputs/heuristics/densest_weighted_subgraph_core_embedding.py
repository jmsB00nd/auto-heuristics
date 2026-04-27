def init_mapping(self):
    import collections
    import heapq

    N = self.num_qubits

    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    interactions = collections.defaultdict(float)
    logical_set = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            if a > b:
                a, b = b, a
            interactions[(a, b)] += 1.0
            logical_set.add(a)
            logical_set.add(b)

    if not interactions:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    log_adj = collections.defaultdict(dict)
    for (a, b), w in interactions.items():
        log_adj[a][b] = w
        log_adj[b][a] = w

    def densest_weighted_subgraph(adj_in):
        adj = {u: dict(nb) for u, nb in adj_in.items()}
        deg = {u: sum(nb.values()) for u, nb in adj.items()}
        nodes = set(adj.keys())
        cur_total = sum(deg.values()) / 2.0
        cur_n = len(nodes)
        best_density = (cur_total / cur_n) if cur_n > 0 else -1.0
        best_set = set(nodes)
        heap = [(deg[u], u) for u in adj]
        heapq.heapify(heap)
        while nodes:
            u = None
            while heap:
                d, cand = heapq.heappop(heap)
                if cand in nodes and abs(deg[cand] - d) < 1e-9:
                    u = cand
                    break
            if u is None:
                break
            for v, w in list(adj[u].items()):
                if v in nodes:
                    deg[v] -= w
                    heapq.heappush(heap, (deg[v], v))
                    cur_total -= w
                    del adj[v][u]
            nodes.remove(u)
            del adj[u]
            cur_n -= 1
            if cur_n > 0:
                density = cur_total / cur_n
                if density > best_density:
                    best_density = density
                    best_set = set(nodes)
        return best_set

    core_logical = densest_weighted_subgraph(log_adj)
    if not core_logical:
        core_logical = set(logical_set)

    phys_adj = collections.defaultdict(set)
    for p in range(N):
        try:
            for q in self.backend[p]:
                if q != p and 0 <= q < N:
                    phys_adj[p].add(q)
        except Exception:
            pass

    K = len(core_logical)
    phys_deg = {p: len(phys_adj[p]) for p in range(N)}

    def densest_connected_subgraph_of_size(k):
        if k <= 0:
            return set()
        seeds = sorted(range(N), key=lambda p: -phys_deg[p])[:min(N, 8)]
        best_set = None
        best_edges = -1
        for seed in seeds:
            chosen = {seed}
            frontier = set(phys_adj[seed])
            while len(chosen) < k and frontier:
                best_v = None
                best_score = -1
                for v in frontier:
                    sc = sum(1 for u in phys_adj[v] if u in chosen)
                    score = sc * (N + 1) + phys_deg[v]
                    if score > best_score:
                        best_score = score
                        best_v = v
                if best_v is None:
                    break
                chosen.add(best_v)
                frontier.discard(best_v)
                for nb in phys_adj[best_v]:
                    if nb not in chosen:
                        frontier.add(nb)
            while len(chosen) < k:
                remaining = [p for p in range(N) if p not in chosen]
                if not remaining:
                    break
                best_p = None
                best_d = float('inf')
                for p in remaining:
                    try:
                        d = min(self.distance_matrix[p][c] for c in chosen)
                    except Exception:
                        d = N
                    if d < best_d:
                        best_d = d
                        best_p = p
                if best_p is None:
                    break
                chosen.add(best_p)
            edges = 0
            for u in chosen:
                for v in phys_adj[u]:
                    if v in chosen and v > u:
                        edges += 1
            if edges > best_edges:
                best_edges = edges
                best_set = set(chosen)
        if best_set is None:
            best_set = set(range(min(k, N)))
        return best_set

    core_physical = densest_connected_subgraph_of_size(K)

    placed_log_to_phys = {}
    placed_phys = set()

    log_wdeg = {u: sum(log_adj[u].values()) for u in core_logical}
    log_order = sorted(core_logical, key=lambda u: -log_wdeg[u])

    phys_in_core_deg = {p: sum(1 for q in phys_adj[p] if q in core_physical) for p in core_physical}

    if log_order and core_physical:
        first_log = log_order[0]
        first_phys = max(core_physical, key=lambda p: (phys_in_core_deg.get(p, 0), phys_deg[p]))
        placed_log_to_phys[first_log] = first_phys
        placed_phys.add(first_phys)

    def attraction_score_phys(log_q, p):
        score = 0.0
        has = False
        for nb_log, w in log_adj[log_q].items():
            if nb_log in placed_log_to_phys:
                has = True
                p2 = placed_log_to_phys[nb_log]
                try:
                    d = self.distance_matrix[p][p2]
                except Exception:
                    d = N
                if d <= 0:
                    continue
                score += w / d
        if not has:
            score = phys_deg[p] * 1e-9
        return score

    for log_q in log_order[1:]:
        candidates = [p for p in core_physical if p not in placed_phys]
        if not candidates:
            candidates = [p for p in range(N) if p not in placed_phys]
        if not candidates:
            break
        best_p = max(candidates, key=lambda p: attraction_score_phys(log_q, p))
        placed_log_to_phys[log_q] = best_p
        placed_phys.add(best_p)

    remaining_logical = [u for u in logical_set if u not in placed_log_to_phys]

    def attraction_to_placed(u):
        s = 0.0
        for v, w in log_adj[u].items():
            if v in placed_log_to_phys:
                s += w
        return s

    while remaining_logical:
        remaining_logical.sort(key=lambda u: -attraction_to_placed(u))
        u = remaining_logical.pop(0)
        candidates = [p for p in range(N) if p not in placed_phys]
        if not candidates:
            break
        best_p = max(candidates, key=lambda p: attraction_score_phys(u, p))
        placed_log_to_phys[u] = best_p
        placed_phys.add(best_p)

    new_map = [-1] * N
    used_phys = set()
    for log_q, phys_q in placed_log_to_phys.items():
        if 0 <= log_q < N and 0 <= phys_q < N and phys_q not in used_phys and new_map[log_q] == -1:
            new_map[log_q] = phys_q
            used_phys.add(phys_q)

    available = [p for p in range(N) if p not in used_phys]
    for log_q in range(N):
        if new_map[log_q] == -1:
            if available:
                new_map[log_q] = available.pop(0)
            else:
                new_map[log_q] = log_q

    self.mapping_dict = new_map
    self.reverse_mapping_dict = [0] * N
    for log_q, phys_q in enumerate(new_map):
        self.reverse_mapping_dict[phys_q] = log_q

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)