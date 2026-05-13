def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    N = int(self.num_qubits)
    if N <= 0:
        self.mapping_dict = []
        self.reverse_mapping_dict = []
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    interactions = {}
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = int(qubits[0]), int(qubits[1])
            if a == b or a >= N or b >= N:
                continue
            key = (a, b) if a < b else (b, a)
            interactions[key] = interactions.get(key, 0) + 1

    dist = self.distance_matrix

    def cost(perm):
        c = 0.0
        for (a, b), w in interactions.items():
            c += w * dist[perm[a]][perm[b]]
        return c

    def sinkhorn(M, iters=15):
        M = np.clip(M, 1e-12, None)
        for _ in range(iters):
            M = M / M.sum(axis=1, keepdims=True)
            M = M / M.sum(axis=0, keepdims=True)
        return M

    def sample_perm(P, rng):
        eps = 1e-12
        logP = np.log(np.clip(P, eps, None))
        u = rng.uniform(eps, 1.0 - eps, size=P.shape)
        gumbel = -np.log(-np.log(u))
        score = logP + gumbel
        row_ind, col_ind = linear_sum_assignment(-score)
        perm = np.empty(N, dtype=np.int64)
        perm[row_ind] = col_ind
        return perm

    rng = np.random.default_rng(12345)

    P = np.full((N, N), 1.0 / N, dtype=np.float64)

    K = 40 if N <= 64 else 24
    elite_frac = 0.2
    n_elite = max(1, int(K * elite_frac))
    n_iters = 25 if N <= 64 else 15
    alpha = 0.5

    best_perm = np.arange(N, dtype=np.int64)
    best_cost = cost(best_perm)

    for _ in range(n_iters):
        samples = []
        costs = []
        for _k in range(K):
            p = sample_perm(P, rng)
            samples.append(p)
            costs.append(cost(p))
        order = np.argsort(costs)
        if costs[order[0]] < best_cost:
            best_cost = costs[order[0]]
            best_perm = samples[order[0]].copy()

        P_emp = np.zeros((N, N), dtype=np.float64)
        for idx in order[:n_elite]:
            e = samples[idx]
            P_emp[np.arange(N), e] += 1.0
        P_emp /= n_elite

        P = (1.0 - alpha) * P + alpha * P_emp
        P = sinkhorn(P, iters=10)

        if float(np.min(P.max(axis=1))) > 0.95:
            break

    eps = 1e-12
    row_ind, col_ind = linear_sum_assignment(-np.log(np.clip(P, eps, None)))
    final_perm = np.empty(N, dtype=np.int64)
    final_perm[row_ind] = col_ind
    if cost(final_perm) > best_cost:
        final_perm = best_perm

    self.mapping_dict = [int(x) for x in final_perm]
    self.reverse_mapping_dict = [0] * N
    for l, p in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)