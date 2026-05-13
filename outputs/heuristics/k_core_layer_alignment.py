def init_mapping(self):
    import networkx as nx

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # --- 1. Build logical interaction graph from self.access ---
    L = nx.Graph()
    active_logicals = set()
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            active_logicals.add(a)
            active_logicals.add(b)
            if L.has_edge(a, b):
                L[a][b]["weight"] += 1
            else:
                L.add_edge(a, b, weight=1)
    for q in active_logicals:
        if q not in L:
            L.add_node(q)

    # --- 2. Build physical coupling graph from self.backend ---
    P = nx.Graph()
    P.add_nodes_from(range(N))
    for u, neighbors in self.backend.items():
        for v in neighbors:
            if u != v:
                P.add_edge(u, v)

    # --- 3. k-core decomposition (drop self-loops first) ---
    L.remove_edges_from(nx.selfloop_edges(L))
    P.remove_edges_from(nx.selfloop_edges(P))
    try:
        log_core = nx.core_number(L) if L.number_of_nodes() > 0 else {}
    except Exception:
        log_core = {q: 0 for q in active_logicals}
    try:
        phys_core = nx.core_number(P)
    except Exception:
        phys_core = {p: 0 for p in range(N)}

    # --- 4. Sort logicals by (core desc, activity desc, id asc) ---
    def _act(q):
        try:
            return self.logical_activity[q]
        except Exception:
            return 0
    sorted_logicals = sorted(
        active_logicals,
        key=lambda q: (-log_core.get(q, 0), -_act(q), q),
    )

    # --- 5. Sort physicals by (core desc, centrality desc, id asc) ---
    def _cent(p):
        try:
            return self.physical_centrality.get(p, 0.0)
        except Exception:
            return 0.0
    sorted_physicals = sorted(
        range(N),
        key=lambda p: (-phys_core.get(p, 0), -_cent(p), p),
    )

    # --- 6. Lockstep alignment: deepest core -> deepest core ---
    used_phys = set()
    assigned_log = set()
    pi = 0
    for lq in sorted_logicals:
        if lq < 0 or lq >= N:
            continue
        while pi < N and sorted_physicals[pi] in used_phys:
            pi += 1
        if pi >= N:
            break
        pq = sorted_physicals[pi]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_phys.add(pq)
        assigned_log.add(lq)
        pi += 1

    # --- 7. Back-fill remaining logicals (identity preferred) ---
    for lq in range(N):
        if self.mapping_dict[lq] != -1:
            continue
        if lq not in used_phys:
            self.mapping_dict[lq] = lq
            self.reverse_mapping_dict[lq] = lq
            used_phys.add(lq)
    free_phys = [p for p in range(N) if p not in used_phys]
    fi = 0
    for lq in range(N):
        if self.mapping_dict[lq] == -1:
            pq = free_phys[fi]
            fi += 1
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)