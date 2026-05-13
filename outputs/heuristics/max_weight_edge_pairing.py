def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # --- 1. QIG edges (weighted, undirected) ---
    qig_edges = []
    seen = set()
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig:
        for u in list(qig.keys()):
            for v, w in qig[u].items():
                if u == v or w <= 0:
                    continue
                key = (u, v) if u < v else (v, u)
                if key in seen:
                    continue
                seen.add(key)
                qig_edges.append((float(w), key[0], key[1]))
    if not qig_edges:
        edge_w = {}
        for gid, qs in self.access.items():
            if len(qs) == 2 and qs[0] != qs[1]:
                a, b = (qs[0], qs[1]) if qs[0] < qs[1] else (qs[1], qs[0])
                edge_w[(a, b)] = edge_w.get((a, b), 0) + 1
        qig_edges = [(float(w), u, v) for (u, v), w in edge_w.items()]
    qig_edges.sort(key=lambda x: -x[0])

    # --- hardware edges (undirected, dedup) ---
    hw_edges = []
    hw_seen = set()
    for pair in self.backend_connections:
        a, b = pair[0], pair[1]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in hw_seen:
            continue
        hw_seen.add(key)
        hw_edges.append(key)

    cent = getattr(self, "physical_centrality", {}) or {}
    def cscore(p, q):
        return float(cent.get(p, 0.0)) + float(cent.get(q, 0.0))
    hw_edges.sort(key=lambda e: -cscore(e[0], e[1]))

    logical_to_phys = {}
    phys_to_logical = {}

    def try_assign(L, P):
        if L in logical_to_phys:
            return logical_to_phys[L] == P
        if P in phys_to_logical:
            return False
        logical_to_phys[L] = P
        phys_to_logical[P] = L
        return True

    # --- 2-4. Hungarian matching and orientation-aware propagation ---
    if qig_edges and hw_edges:
        n_qig, n_hw = len(qig_edges), len(hw_edges)
        size = max(n_qig, n_hw)
        cost = np.zeros((size, size), dtype=float)
        for i, (w, _u, _v) in enumerate(qig_edges):
            for j, (p, q) in enumerate(hw_edges):
                cost[i, j] = -(w * cscore(p, q))
        try:
            row_ind, col_ind = linear_sum_assignment(cost)
        except Exception:
            row_ind = list(range(size))
            col_ind = list(range(size))

        matches = []
        for r, c in zip(row_ind, col_ind):
            if r < n_qig and c < n_hw:
                matches.append((qig_edges[r], hw_edges[c]))
        matches.sort(key=lambda m: -m[0][0])

        act = getattr(self, "logical_activity", {}) or {}
        for (w, u, v), (p, q) in matches:
            uf, vf = logical_to_phys.get(u), logical_to_phys.get(v)
            if uf is not None and vf is not None:
                continue
            pf, qf = phys_to_logical.get(p), phys_to_logical.get(q)
            ok1 = (uf in (None, p)) and (vf in (None, q)) and (pf in (None, u)) and (qf in (None, v))
            ok2 = (uf in (None, q)) and (vf in (None, p)) and (qf in (None, u)) and (pf in (None, v))
            if ok1 and ok2:
                au, av = float(act.get(u, 0)), float(act.get(v, 0))
                cp, cq = float(cent.get(p, 0.0)), float(cent.get(q, 0.0))
                if au * cp + av * cq >= au * cq + av * cp:
                    try_assign(u, p); try_assign(v, q)
                else:
                    try_assign(u, q); try_assign(v, p)
            elif ok1:
                try_assign(u, p); try_assign(v, q)
            elif ok2:
                try_assign(u, q); try_assign(v, p)
            else:
                free_phys = [x for x in (p, q) if x not in phys_to_logical]
                for L in (u, v):
                    if L not in logical_to_phys and free_phys:
                        best = max(free_phys, key=lambda x: float(cent.get(x, 0.0)))
                        try_assign(L, best)
                        free_phys.remove(best)

    # --- 5. Back-fill logicals appearing in access ---
    act = getattr(self, "logical_activity", {}) or {}
    all_logicals = set()
    for _gid, qs in self.access.items():
        for q in qs:
            if isinstance(q, int) and 0 <= q < N:
                all_logicals.add(q)

    unassigned_logicals = sorted(
        [L for L in all_logicals if L not in logical_to_phys],
        key=lambda L: -float(act.get(L, 0)),
    )
    free_phys_sorted = sorted(
        [p for p in range(N) if p not in phys_to_logical],
        key=lambda p: -float(cent.get(p, 0.0)),
    )
    for L in unassigned_logicals:
        if not free_phys_sorted:
            break
        P = free_phys_sorted.pop(0)
        logical_to_phys[L] = P
        phys_to_logical[P] = L

    # --- 6. Identity-style fallback for leftover logicals/physicals ---
    remaining_logicals = [L for L in range(N) if L not in logical_to_phys]
    remaining_phys = [P for P in range(N) if P not in phys_to_logical]
    for L, P in zip(remaining_logicals, remaining_phys):
        logical_to_phys[L] = P
        phys_to_logical[P] = L

    for L in range(N):
        P = logical_to_phys.get(L, L)
        self.mapping_dict[L] = P
    for P in range(N):
        self.reverse_mapping_dict[P] = -1
    for L in range(N):
        self.reverse_mapping_dict[self.mapping_dict[L]] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)