def init_mapping(self):
    import numpy as np

    N = int(self.num_qubits)

    if N == 0:
        self.mapping_dict = []
        self.reverse_mapping_dict = []
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    def _greedy_assign(cost):
        n = cost.shape[0]
        used = [False] * n
        col = [0] * n
        order = list(np.argsort(cost.min(axis=1)))
        for r in order:
            best_c, best_v = -1, float("inf")
            for c in range(n):
                if used[c]:
                    continue
                v = float(cost[r, c])
                if v < best_v:
                    best_v, best_c = v, c
            if best_c < 0:
                for c in range(n):
                    if not used[c]:
                        best_c = c
                        break
            col[r] = best_c
            used[best_c] = True
        return list(range(n)), col

    try:
        from scipy.optimize import linear_sum_assignment

        def _assign(M):
            r, c = linear_sum_assignment(M)
            return list(r), list(c)
    except Exception:
        def _assign(M):
            return _greedy_assign(M)

    W = np.zeros((N, N), dtype=float)
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = int(qubits[0]), int(qubits[1])
            if 0 <= q1 < N and 0 <= q2 < N and q1 != q2:
                W[q1, q2] += 1.0
                W[q2, q1] += 1.0

    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            try:
                D[i, j] = float(self.distance_matrix[i][j])
            except Exception:
                D[i, j] = 0.0

    if W.sum() == 0.0:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    X = np.full((N, N), 1.0 / N, dtype=float)

    max_iter = max(10, min(50, 3 * N))
    for it in range(max_iter):
        G = W @ X @ D
        if not np.all(np.isfinite(G)):
            break
        r_idx, c_idx = _assign(G)
        S = np.zeros((N, N), dtype=float)
        for ri, ci in zip(r_idx, c_idx):
            S[ri, ci] = 1.0
        gamma = 2.0 / (it + 2.0)
        X = (1.0 - gamma) * X + gamma * S

    Xs = np.maximum(X, 1e-12)
    for _ in range(50):
        rs = Xs.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        Xs = Xs / rs
        cs = Xs.sum(axis=0, keepdims=True)
        cs[cs == 0] = 1.0
        Xs = Xs / cs

    if not np.all(np.isfinite(Xs)):
        Xs = np.full((N, N), 1.0 / N, dtype=float)

    r_idx, c_idx = _assign(-Xs)

    mapping = [-1] * N
    reverse = [-1] * N
    used_phys = [False] * N
    for ri, ci in zip(r_idx, c_idx):
        ri, ci = int(ri), int(ci)
        if 0 <= ri < N and 0 <= ci < N and not used_phys[ci] and mapping[ri] == -1:
            mapping[ri] = ci
            reverse[ci] = ri
            used_phys[ci] = True

    free_phys = [p for p in range(N) if not used_phys[p]]
    fp_iter = iter(free_phys)
    for l in range(N):
        if mapping[l] == -1:
            try:
                p = next(fp_iter)
            except StopIteration:
                p = l
            mapping[l] = p
            reverse[p] = l

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)