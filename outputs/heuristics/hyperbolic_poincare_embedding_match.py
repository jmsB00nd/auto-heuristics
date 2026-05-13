def init_mapping(self):
    import numpy as np
    import math
    from collections import deque, defaultdict

    N = self.num_qubits
    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    try:
        logical_set = set()
        edges = []
        for _gid, qubits in self.access.items():
            for q in qubits:
                if 0 <= q < N:
                    logical_set.add(q)
            if len(qubits) == 2 and qubits[0] != qubits[1]:
                a, b = qubits[0], qubits[1]
                if 0 <= a < N and 0 <= b < N:
                    edges.append((a, b))

        if not logical_set:
            assert len(set(self.mapping_dict)) == len(self.mapping_dict)
            return

        logical_list = sorted(logical_set)
        L = len(logical_list)
        log_idx = {q: i for i, q in enumerate(logical_list)}

        qig_adj = defaultdict(set)
        for q1, q2 in edges:
            qig_adj[log_idx[q1]].add(log_idx[q2])
            qig_adj[log_idx[q2]].add(log_idx[q1])

        qig_dist = np.full((L, L), -1.0)
        for i in range(L):
            qig_dist[i, i] = 0.0
            visited = {i: 0}
            queue = deque([i])
            while queue:
                u = queue.popleft()
                for v in qig_adj[u]:
                    if v not in visited:
                        visited[v] = visited[u] + 1
                        queue.append(v)
            for v, d in visited.items():
                qig_dist[i, v] = d
        finite_max = qig_dist[qig_dist >= 0].max() if (qig_dist >= 0).any() else 1.0
        qig_dist[qig_dist < 0] = finite_max + 1.0

        hw_dist = np.array(self.distance_matrix, dtype=float)
        if hw_dist.shape[0] != N:
            hw_dist = np.zeros((N, N))

        def embed_poincare(D, n_iter, lr, seed):
            rng = np.random.RandomState(seed)
            n = D.shape[0]
            if n <= 1:
                return np.zeros((n, 2))
            X = rng.randn(n, 2) * 0.05
            for _ in range(n_iter):
                sq_norm = np.sum(X * X, axis=1)
                sq_norm = np.minimum(sq_norm, 1.0 - 1e-6)
                alpha = 1.0 - sq_norm
                diff = X[:, None, :] - X[None, :, :]
                sq_diff = np.sum(diff * diff, axis=-1)
                beta = alpha[:, None] * alpha[None, :] + 1e-12
                gamma = 1.0 + 2.0 * sq_diff / beta
                gamma = np.maximum(gamma, 1.0 + 1e-10)
                d = np.arccosh(gamma)
                np.fill_diagonal(d, 0.0)
                err = d - D
                np.fill_diagonal(err, 0.0)
                denom1 = beta[:, :, None]
                denom2 = (alpha[:, None] ** 2 * alpha[None, :])[:, :, None] + 1e-12
                term1 = 4.0 * diff / denom1
                term2 = 4.0 * sq_diff[:, :, None] * X[:, None, :] / denom2
                dgamma = term1 + term2
                sqrt_g = np.sqrt(np.maximum(gamma * gamma - 1.0, 1e-12))
                dd = dgamma / sqrt_g[:, :, None]
                gpp = 2.0 * err[:, :, None] * dd
                for k in range(2):
                    np.fill_diagonal(gpp[:, :, k], 0.0)
                grad = np.sum(gpp, axis=1)
                rgrad = (alpha[:, None] ** 2 / 4.0) * grad
                X = X - lr * rgrad
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                cap = 1.0 - 1e-5
                X = np.where(norms > cap, X * cap / (norms + 1e-12), X)
            return X

        s_qig = 2.0 / max(qig_dist.max(), 1.0)
        s_hw = 2.0 / max(hw_dist.max(), 1.0)
        n_iter = 80 if max(N, L) <= 64 else 40
        X_qig = embed_poincare(qig_dist * s_qig, n_iter=n_iter, lr=0.03, seed=0)
        X_hw = embed_poincare(hw_dist * s_hw, n_iter=n_iter, lr=0.03, seed=1)

        qig_score = np.array([float(self.logical_activity.get(logical_list[i], 0)) for i in range(L)])
        hw_score = np.array([float(self.physical_centrality.get(p, 0.0)) for p in range(N)])

        ang_qig = 0.0
        if L > 0 and qig_score.max() > 0:
            i_a = int(np.argmax(qig_score))
            ang_qig = math.atan2(X_qig[i_a, 1], X_qig[i_a, 0])
        ang_hw = 0.0
        if N > 0 and hw_score.max() > 0:
            p_a = int(np.argmax(hw_score))
            ang_hw = math.atan2(X_hw[p_a, 1], X_hw[p_a, 0])
        rot = ang_qig - ang_hw
        cR, sR = math.cos(rot), math.sin(rot)
        R = np.array([[cR, -sR], [sR, cR]])
        X_hw_a = X_hw @ R.T

        sq_q = np.sum(X_qig * X_qig, axis=1)
        sq_h = np.sum(X_hw_a * X_hw_a, axis=1)
        diff_qh = X_qig[:, None, :] - X_hw_a[None, :, :]
        sq_d = np.sum(diff_qh * diff_qh, axis=-1)
        denom = (1.0 - sq_q[:, None]) * (1.0 - sq_h[None, :]) + 1e-12
        arg = np.maximum(1.0 + 2.0 * sq_d / denom, 1.0)
        cost = np.arccosh(arg)

        max_c = float(cost.max()) if cost.size > 0 else 1.0
        big_cost = np.full((N, N), max_c)
        for i, lq in enumerate(logical_list):
            big_cost[lq, :] = cost[i, :]

        mapping = [-1] * N
        reverse = [-1] * N
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(big_cost)
            for r, p in zip(row_ind, col_ind):
                mapping[int(r)] = int(p)
                reverse[int(p)] = int(r)
        except Exception:
            used = set()
            order = sorted(range(L), key=lambda i: -qig_score[i])
            for i in order:
                lq = logical_list[i]
                best_p, best_c = -1, float('inf')
                for p in range(N):
                    if p in used:
                        continue
                    if cost[i, p] < best_c:
                        best_c = cost[i, p]
                        best_p = p
                if best_p >= 0:
                    mapping[lq] = best_p
                    reverse[best_p] = lq
                    used.add(best_p)
            unused = [p for p in range(N) if p not in used]
            unassigned = [q for q in range(N) if mapping[q] == -1]
            for lq, pq in zip(unassigned, unused):
                mapping[lq] = pq
                reverse[pq] = lq

        if -1 not in mapping and len(set(mapping)) == N:
            self.mapping_dict = mapping
            self.reverse_mapping_dict = reverse
        else:
            self.mapping_dict = list(range(N))
            self.reverse_mapping_dict = list(range(N))
    except Exception:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)