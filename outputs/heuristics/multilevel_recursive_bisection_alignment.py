def init_mapping(self):
    import numpy as np

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    log_edges = {}
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            log_edges[key] = log_edges.get(key, 0.0) + 1.0

    phys_edges = {}
    for p in range(N):
        for q in self.backend.get(p, set()):
            if p < q:
                phys_edges[(p, q)] = 1.0

    def laplacian(nodes, edges_w):
        idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        L = np.zeros((n, n), dtype=float)
        for (a, b), w in edges_w.items():
            if a in idx and b in idx:
                i, j = idx[a], idx[b]
                L[i, j] -= w
                L[j, i] -= w
                L[i, i] += w
                L[j, j] += w
        return L

    def fiedler_order(nodes, edges_w):
        if len(nodes) <= 1:
            return list(nodes)
        try:
            L = laplacian(nodes, edges_w)
            L = L + 1e-9 * np.eye(len(nodes))
            vals, vecs = np.linalg.eigh(L)
            v = vecs[:, 1] if len(nodes) >= 2 else vecs[:, 0]
            order = np.argsort(v)
            return [nodes[i] for i in order]
        except Exception:
            return list(nodes)

    def density(nodes, edges_w):
        s = 0.0
        nset = set(nodes)
        for (a, b), w in edges_w.items():
            if a in nset and b in nset:
                s += w
        return s

    def phys_conn(nodes, edges_w):
        s = density(nodes, edges_w)
        s += sum(self.physical_centrality.get(p, 0.0) for p in nodes)
        return s

    def recurse(log_nodes, phys_nodes):
        k = len(log_nodes)
        if k == 0:
            return
        if k == 1:
            l = log_nodes[0]
            p = phys_nodes[0]
            self.mapping_dict[l] = p
            self.reverse_mapping_dict[p] = l
            return
        log_order = fiedler_order(log_nodes, log_edges)
        phys_order = fiedler_order(phys_nodes, phys_edges)
        mid = k // 2
        la = log_order[:mid]
        lb = log_order[mid:]
        pa = phys_order[:mid]
        pb = phys_order[mid:]
        d_la = density(la, log_edges)
        d_lb = density(lb, log_edges)
        c_pa = phys_conn(pa, phys_edges)
        c_pb = phys_conn(pb, phys_edges)
        if (d_la >= d_lb) == (c_pa >= c_pb):
            recurse(la, pa)
            recurse(lb, pb)
        else:
            recurse(la, pb)
            recurse(lb, pa)

    try:
        recurse(list(range(N)), list(range(N)))
    except Exception:
        pass

    used_phys = set(p for p in self.mapping_dict if p != -1)
    free_phys = [p for p in range(N) if p not in used_phys]
    free_log = [l for l in range(N) if self.mapping_dict[l] == -1]
    for l, p in zip(free_log, free_phys):
        self.mapping_dict[l] = p
        self.reverse_mapping_dict[p] = l

    used_phys = set(p for p in self.mapping_dict if p != -1)
    remaining_phys_iter = iter(p for p in range(N) if p not in used_phys)
    for l in range(N):
        if self.mapping_dict[l] == -1:
            try:
                p = next(remaining_phys_iter)
            except StopIteration:
                break
            self.mapping_dict[l] = p
            self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)