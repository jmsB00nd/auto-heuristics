def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    interactions = []
    logical_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a != b:
                interactions.append((a, b))
            logical_nodes.add(a)
            logical_nodes.add(b)

    G_log = nx.Graph()
    G_log.add_nodes_from(range(N))
    G_log.add_edges_from(interactions)

    G_phys = nx.Graph()
    G_phys.add_nodes_from(range(N))
    try:
        for u, adj in enumerate(self.backend):
            for v in adj:
                if u != v:
                    G_phys.add_edge(int(u), int(v))
    except Exception:
        for (u, v) in self.backend_connections:
            if u != v:
                G_phys.add_edge(int(u), int(v))

    G_log_sl = G_log.copy()
    G_log_sl.remove_edges_from(nx.selfloop_edges(G_log_sl))
    G_phys_sl = G_phys.copy()
    G_phys_sl.remove_edges_from(nx.selfloop_edges(G_phys_sl))

    try:
        log_core = nx.core_number(G_log_sl)
    except Exception:
        log_core = {v: G_log_sl.degree(v) for v in G_log_sl.nodes()}
    try:
        phys_core = nx.core_number(G_phys_sl)
    except Exception:
        phys_core = {v: G_phys_sl.degree(v) for v in G_phys_sl.nodes()}

    log_shells = defaultdict(list)
    for v in logical_nodes:
        log_shells[log_core.get(v, 0)].append(v)
    phys_shells = defaultdict(list)
    for p in range(N):
        phys_shells[phys_core.get(p, 0)].append(p)

    log_shell_keys = sorted(log_shells.keys(), reverse=True)
    phys_shell_keys = sorted(phys_shells.keys(), reverse=True)

    used_phys = set()
    placed_anchors = []

    def phys_score(p):
        if not placed_anchors:
            return 0.0
        s = 0.0
        for ap in placed_anchors:
            try:
                s += float(self.distance_matrix[p][ap])
            except Exception:
                s += 0.0
        return s

    phys_pool = []
    for sk in phys_shell_keys:
        bucket = sorted(
            phys_shells[sk],
            key=lambda p: (-G_phys_sl.degree(p), p),
        )
        phys_pool.append((sk, bucket))

    def take_phys(prefer_close=True):
        for i, (sk, bucket) in enumerate(phys_pool):
            candidates = [p for p in bucket if p not in used_phys]
            if not candidates:
                continue
            if placed_anchors and prefer_close:
                candidates.sort(key=lambda p: (phys_score(p), -G_phys_sl.degree(p), p))
            chosen = candidates[0]
            phys_pool[i] = (sk, [p for p in bucket if p != chosen])
            return chosen
        return None

    for lk in log_shell_keys:
        log_bucket = sorted(
            log_shells[lk],
            key=lambda v: (-G_log_sl.degree(v), v),
        )
        for lq in log_bucket:
            if self.mapping_dict[lq] is not None:
                continue
            pq = take_phys(prefer_close=True)
            if pq is None:
                break
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_phys.add(pq)
            placed_anchors.append(pq)

    for lq in range(N):
        if self.mapping_dict[lq] is not None:
            continue
        if lq not in used_phys:
            self.mapping_dict[lq] = lq
            self.reverse_mapping_dict[lq] = lq
            used_phys.add(lq)
        else:
            pq = take_phys(prefer_close=False)
            if pq is None:
                for cand in range(N):
                    if cand not in used_phys:
                        pq = cand
                        break
            if pq is not None:
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_phys.add(pq)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)