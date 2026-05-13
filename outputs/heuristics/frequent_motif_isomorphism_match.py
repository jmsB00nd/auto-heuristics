def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    logical_edges = defaultdict(int)
    logical_qubits = set()
    access = getattr(self, "access", {}) or {}
    for _gid, qubits in access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            if q1 == q2 or q1 >= N or q2 >= N:
                continue
            a, b = (q1, q2) if q1 < q2 else (q2, q1)
            logical_edges[(a, b)] += 1
            logical_qubits.add(q1)
            logical_qubits.add(q2)

    adj = defaultdict(set)
    for (a, b) in logical_edges:
        adj[a].add(b)
        adj[b].add(a)

    def ew(a, b):
        if a > b:
            a, b = b, a
        return logical_edges.get((a, b), 0)

    G_phys = nx.Graph()
    G_phys.add_nodes_from(range(N))
    for p in range(N):
        for nbr in self.backend.get(p, set()):
            if p < nbr:
                G_phys.add_edge(p, nbr)

    assigned_logical = set()
    used_physical = set()

    def commit(log_q, phys_q):
        if log_q < 0 or phys_q < 0 or log_q >= N or phys_q >= N:
            return False
        if log_q in assigned_logical or phys_q in used_physical:
            return False
        self.mapping_dict[log_q] = phys_q
        self.reverse_mapping_dict[phys_q] = log_q
        assigned_logical.add(log_q)
        used_physical.add(phys_q)
        return True

    triangles = []
    seen_tri = set()
    for a in logical_qubits:
        for b in adj[a]:
            if b <= a:
                continue
            for c in adj[a] & adj[b]:
                if c <= b:
                    continue
                key = (a, b, c)
                if key in seen_tri:
                    continue
                seen_tri.add(key)
                w = ew(a, b) + ew(a, c) + ew(b, c)
                triangles.append((w, key))
    triangles.sort(reverse=True)

    k4s = []
    seen_k4 = set()
    for _w, (a, b, c) in triangles[:60]:
        common = adj[a] & adj[b] & adj[c]
        for d in common:
            if d <= c:
                continue
            quad = (a, b, c, d)
            if quad in seen_k4:
                continue
            seen_k4.add(quad)
            tw = (ew(a, b) + ew(a, c) + ew(a, d)
                  + ew(b, c) + ew(b, d) + ew(c, d))
            k4s.append((tw, quad))
    k4s.sort(reverse=True)

    stars = []
    for hub in logical_qubits:
        nbrs = list(adj[hub])
        if len(nbrs) >= 3:
            nbrs.sort(key=lambda x: -ew(hub, x))
            top = nbrs[: min(len(nbrs), 4)]
            sw = sum(ew(hub, n) for n in top)
            stars.append((sw, hub, tuple(top)))
    stars.sort(reverse=True)

    def try_anchor(motif_nx, motif_nodes):
        if any(n in assigned_logical for n in motif_nodes):
            return False
        avail = [p for p in range(N) if p not in used_physical]
        if len(avail) < len(motif_nodes):
            return False
        H = G_phys.subgraph(avail).copy()
        try:
            GM = nx.isomorphism.GraphMatcher(H, motif_nx)
            it = GM.subgraph_isomorphisms_iter()
            best = None
            best_score = float("inf")
            tries = 0
            cent = getattr(self, "physical_centrality", {}) or {}
            for iso in it:
                tries += 1
                score = -sum(cent.get(p, 0.0) for p in iso.keys())
                if score < best_score:
                    best_score = score
                    best = iso
                if tries >= 8:
                    break
            if best is None:
                return False
            tentative = {log: phys for phys, log in best.items()}
            if set(tentative.keys()) != set(motif_nodes):
                return False
            if len(set(tentative.values())) != len(tentative):
                return False
            for log, phys in tentative.items():
                if not commit(log, phys):
                    return False
            return True
        except Exception:
            return False

    for _w, (a, b, c) in triangles[:5]:
        m = nx.Graph()
        m.add_edges_from([(a, b), (a, c), (b, c)])
        try_anchor(m, [a, b, c])

    for _w, (a, b, c, d) in k4s[:3]:
        m = nx.Graph()
        m.add_edges_from([(a, b), (a, c), (a, d),
                          (b, c), (b, d), (c, d)])
        try_anchor(m, [a, b, c, d])

    for _sw, hub, leaves in stars[:5]:
        m = nx.Graph()
        for lf in leaves:
            m.add_edge(hub, lf)
        try_anchor(m, [hub] + list(leaves))

    activity = defaultdict(int)
    for (a, b), w in logical_edges.items():
        activity[a] += w
        activity[b] += w
    remaining = sorted(
        [q for q in logical_qubits if q not in assigned_logical],
        key=lambda q: -activity[q],
    )
    cent_map = getattr(self, "physical_centrality", {}) or {}

    for log_q in remaining:
        mapped_nbrs = [n for n in adj[log_q] if n in assigned_logical]
        best_phys = -1
        best_score = float("inf")
        for phys in range(N):
            if phys in used_physical:
                continue
            if mapped_nbrs:
                s = 0.0
                for nb in mapped_nbrs:
                    s += ew(log_q, nb) * self.distance_matrix[phys][self.mapping_dict[nb]]
            else:
                s = -cent_map.get(phys, 0.0) * 1e6
            if s < best_score:
                best_score = s
                best_phys = phys
        if best_phys >= 0:
            commit(log_q, best_phys)

    free_logs = [q for q in range(N) if self.mapping_dict[q] == -1]
    free_phys = [p for p in range(N) if p not in used_physical]
    for log_q, phys_q in zip(free_logs, free_phys):
        commit(log_q, phys_q)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)