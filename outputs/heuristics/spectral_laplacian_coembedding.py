def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    try:
        import scipy.linalg as sla

        # --- Collect active logical qubits and interactions ---
        logical_set = set()
        edge_weights = defaultdict(float)
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                if a == b:
                    continue
                logical_set.add(a)
                logical_set.add(b)
                key = (a, b) if a < b else (b, a)
                w = 1.0
                try:
                    w = float(self.qubit_interaction_graph[a][b]) or 1.0
                except Exception:
                    w = 1.0
                edge_weights[key] = w

        active_logicals = sorted(logical_set)
        n_log = len(active_logicals)

        def normalized_laplacian(n, edges_w):
            A = np.zeros((n, n), dtype=float)
            for (i, j), w in edges_w.items():
                A[i, j] = w
                A[j, i] = w
            d = A.sum(axis=1)
            d_inv_sqrt = np.zeros(n)
            nz = d > 0
            d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
            D_inv_sqrt = np.diag(d_inv_sqrt)
            L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
            L = 0.5 * (L + L.T)
            return L

        def spectral_embed(L, k):
            n = L.shape[0]
            if n <= 1 or k <= 0:
                return np.zeros((n, max(k, 1)))
            try:
                w, V = sla.eigh(L)
            except Exception:
                w, V = np.linalg.eigh(L)
            order = np.argsort(w)
                # drop trivial (smallest) eigenvector
            take = order[1:1 + k] if len(order) > k else order[:k]
            X = V[:, take]
            if X.shape[1] < k:
                pad = np.zeros((n, k - X.shape[1]))
                X = np.hstack([X, pad])
            return X

        # --- Build logical Laplacian on compact index space ---
        log_index = {q: i for i, q in enumerate(active_logicals)}
        log_edges = {}
        for (a, b), w in edge_weights.items():
            if a in log_index and b in log_index:
                log_edges[(log_index[a], log_index[b])] = w

        # --- Build physical Laplacian ---
        phys_edges = {}
        for p, neighbors in self.backend.items():
            for q in neighbors:
                if p == q:
                    continue
                key = (p, q) if p < q else (q, p)
                phys_edges[key] = 1.0

        used_phys = set()

        if n_log >= 2 and N >= 2:
            k = min(n_log - 1, N - 1, 4)
            if k >= 1:
                L_log = normalized_laplacian(n_log, log_edges)
                L_phy = normalized_laplacian(N, phys_edges)
                X_log = spectral_embed(L_log, k)
                X_phy = spectral_embed(L_phy, k)

                # Standardize scales (per dimension)
                def standardize(X):
                    mu = X.mean(axis=0, keepdims=True)
                    Xc = X - mu
                    norms = np.linalg.norm(Xc, axis=0, keepdims=True) + 1e-12
                    return Xc / norms

                Xl = standardize(X_log)
                Xp = standardize(X_phy)

                # Orthogonal Procrustes: rotate Xl to best fit a subset of Xp
                # Use mean as anchor; optimal R from SVD of Xp_subset^T Xl
                # Since clouds differ in size, compute R from full clouds via cross-covariance proxy
                M_cov = Xp[:n_log].T @ Xl if Xp.shape[0] >= n_log else Xp.T @ Xl[:Xp.shape[0]]
                try:
                    U, _, Vt = np.linalg.svd(M_cov)
                    R = U @ Vt
                except Exception:
                    R = np.eye(Xl.shape[1])
                Xl_aligned = Xl @ R.T

                # Nearest-neighbor matching with used sets
                # Distance matrix: n_log x N
                diff = Xl_aligned[:, None, :] - Xp[None, :, :]
                D = np.linalg.norm(diff, axis=2)

                # Greedy smallest-first matching
                pairs = []
                flat_order = np.argsort(D, axis=None)
                used_log_local = set()
                for idx in flat_order:
                    li = idx // N
                    pj = idx % N
                    if li in used_log_local or pj in used_phys:
                        continue
                    used_log_local.add(int(li))
                    used_phys.add(int(pj))
                    pairs.append((active_logicals[int(li)], int(pj)))
                    if len(used_log_local) == n_log:
                        break

                for log_q, phy_q in pairs:
                    self.mapping_dict[log_q] = phy_q
                    self.reverse_mapping_dict[phy_q] = log_q

        # --- Fallback: identity-style fill for remaining logical qubits ---
        free_phys = [p for p in range(N) if p not in used_phys]
        fp_iter = iter(free_phys)
        for log_q in range(N):
            if self.mapping_dict[log_q] == -1:
                # prefer identity if available
                if log_q not in used_phys:
                    self.mapping_dict[log_q] = log_q
                    self.reverse_mapping_dict[log_q] = log_q
                    used_phys.add(log_q)
                else:
                    # pick next free physical
                    p = next((x for x in fp_iter if x not in used_phys), None)
                    if p is None:
                        # rebuild iterator from current free set
                        remaining = [x for x in range(N) if x not in used_phys]
                        if remaining:
                            p = remaining[0]
                    if p is not None:
                        self.mapping_dict[log_q] = p
                        self.reverse_mapping_dict[p] = log_q
                        used_phys.add(p)

    except Exception:
        # Safe fallback: identity
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    # Final safety: ensure no -1 remains and bijection holds
    if -1 in self.mapping_dict or len(set(self.mapping_dict)) != N:
        used = set()
        new_map = [-1] * N
        for i, p in enumerate(self.mapping_dict):
            if p != -1 and p not in used:
                new_map[i] = p
                used.add(p)
        free = [p for p in range(N) if p not in used]
        fi = 0
        for i in range(N):
            if new_map[i] == -1:
                new_map[i] = free[fi]
                fi += 1
        self.mapping_dict = new_map
        self.reverse_mapping_dict = [-1] * N
        for l, p in enumerate(self.mapping_dict):
            self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)