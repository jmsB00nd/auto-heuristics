def init_mapping(self):
    import numpy as np
    N = self.num_qubits

    self.mapping_dict = [i for i in range(N)]
    self.reverse_mapping_dict = [i for i in range(N)]

    try:
        from scipy.optimize import linear_sum_assignment
        try:
            from scipy.linalg import eigh as _eigh
        except Exception:
            from numpy.linalg import eigh as _eigh

        def build_laplacian_from_edges(n, edges_with_weight):
            A = np.zeros((n, n), dtype=float)
            for u, v, w in edges_with_weight:
                if u == v or u < 0 or v < 0 or u >= n or v >= n:
                    continue
                A[u, v] += w
                A[v, u] += w
            d = A.sum(axis=1)
            L = np.diag(d) - A
            return L

        log_edges = []
        if hasattr(self, "qubit_interaction_graph") and self.qubit_interaction_graph is not None:
            seen = set()
            for u, nbrs in self.qubit_interaction_graph.items():
                for v, w in nbrs.items():
                    if u == v:
                        continue
                    key = (u, v) if u < v else (v, u)
                    if key in seen:
                        continue
                    seen.add(key)
                    log_edges.append((key[0], key[1], float(w)))
        else:
            counts = {}
            for _gid, qs in self.access.items():
                if len(qs) == 2:
                    a, b = qs[0], qs[1]
                    if a == b:
                        continue
                    key = (a, b) if a < b else (b, a)
                    counts[key] = counts.get(key, 0) + 1
            for (a, b), w in counts.items():
                log_edges.append((a, b, float(w)))

        hw_edges = []
        seen_hw = set()
        for u, nbrs in self.backend.items():
            for v in nbrs:
                if u == v:
                    continue
                key = (u, v) if u < v else (v, u)
                if key in seen_hw:
                    continue
                seen_hw.add(key)
                hw_edges.append((key[0], key[1], 1.0))

        L_log = build_laplacian_from_edges(N, log_edges)
        L_hw = build_laplacian_from_edges(N, hw_edges)

        L_log = 0.5 * (L_log + L_log.T)
        L_hw = 0.5 * (L_hw + L_hw.T)

        w_log, V_log = _eigh(L_log)
        w_hw, V_hw = _eigh(L_hw)

        times = np.array([0.1, 1.0, 10.0], dtype=float)

        def hks(eigvals, eigvecs, ts):
            V2 = eigvecs ** 2
            E = np.exp(-np.outer(eigvals, ts))
            return V2 @ E

        H_log = hks(w_log, V_log, times)
        H_hw = hks(w_hw, V_hw, times)

        active = set()
        for _gid, qs in self.access.items():
            for q in qs:
                if 0 <= q < N:
                    active.add(q)

        if len(active) > 0:
            scale = max(1e-12, float(np.max(np.abs(H_log)) + np.max(np.abs(H_hw))))
            H_log = H_log / scale
            H_hw = H_hw / scale

        diff = H_log[:, None, :] - H_hw[None, :, :]
        cost = np.sqrt(np.sum(diff * diff, axis=2))

        for L in range(N):
            if L not in active:
                cost[L, :] = 0.0

        row_ind, col_ind = linear_sum_assignment(cost)

        mapping = [0] * N
        reverse = [0] * N
        for r, c in zip(row_ind, col_ind):
            mapping[r] = int(c)
            reverse[int(c)] = r

        self.mapping_dict = mapping
        self.reverse_mapping_dict = reverse
    except Exception:
        self.mapping_dict = [i for i in range(N)]
        self.reverse_mapping_dict = [i for i in range(N)]

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)