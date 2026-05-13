def init_mapping(self):
    import numpy as np

    N = int(self.num_qubits)

    # ---- Logical weighted adjacency from 2-qubit gates in self.access ----
    A_log = np.zeros((N, N), dtype=float)
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            if 0 <= q1 < N and 0 <= q2 < N and q1 != q2:
                A_log[q1, q2] += 1.0
                A_log[q2, q1] += 1.0

    # ---- Physical adjacency from coupling graph ----
    A_phys = np.zeros((N, N), dtype=float)
    for u, neighbors in self.backend.items():
        if 0 <= u < N:
            for v in neighbors:
                if 0 <= v < N and u != v:
                    A_phys[u, v] = 1.0
                    A_phys[v, u] = 1.0

    def compute_ppr_matrix(A, alpha=0.15):
        n = A.shape[0]
        d = A.sum(axis=1)
        P = np.zeros_like(A)
        nz = d > 0
        if np.any(nz):
            P[nz] = A[nz] / d[nz, None]
        # self-loops for isolated nodes -> stationary at themselves
        iso = np.where(~nz)[0]
        for i in iso:
            P[i, i] = 1.0
        I = np.eye(n)
        Mat = I - (1.0 - alpha) * P.T
        try:
            M = np.linalg.solve(Mat, alpha * I)
        except np.linalg.LinAlgError:
            M = alpha * np.linalg.pinv(Mat)
        return M

    try:
        PPR_log = compute_ppr_matrix(A_log)
        PPR_phys = compute_ppr_matrix(A_phys)

        # Permutation-invariant projection: sort each fingerprint descending
        fp_log = -np.sort(-PPR_log, axis=0)
        fp_phys = -np.sort(-PPR_phys, axis=0)

        # Pairwise Euclidean distance via (||a||^2 + ||b||^2 - 2 a.b)
        log_sq = np.sum(fp_log * fp_log, axis=0)
        phys_sq = np.sum(fp_phys * fp_phys, axis=0)
        cross = fp_log.T @ fp_phys
        cost_sq = log_sq[:, None] + phys_sq[None, :] - 2.0 * cross
        cost = np.sqrt(np.maximum(cost_sq, 0.0))

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        mapping = [0] * N
        reverse = [0] * N
        for log_q, phys_q in zip(row_ind, col_ind):
            mapping[int(log_q)] = int(phys_q)
            reverse[int(phys_q)] = int(log_q)

        self.mapping_dict = mapping
        self.reverse_mapping_dict = reverse
    except Exception:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)