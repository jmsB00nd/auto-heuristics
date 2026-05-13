def init_mapping(self):
    N = self.num_qubits
    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        W = np.zeros((N, N), dtype=float)
        qig = getattr(self, "qubit_interaction_graph", None)
        if qig is not None:
            for q1, neigh in qig.items():
                if q1 >= N:
                    continue
                for q2, w in neigh.items():
                    if q2 >= N or q1 == q2:
                        continue
                    W[q1, q2] = float(w)
        else:
            for gate_id, qubits in self.access.items():
                if len(qubits) == 2:
                    a, b = qubits[0], qubits[1]
                    if a < N and b < N and a != b:
                        W[a, b] += 1.0
                        W[b, a] += 1.0

        W = 0.5 * (W + W.T)

        D = np.zeros((N, N), dtype=float)
        for i in range(N):
            row = self.distance_matrix[i]
            for j in range(N):
                D[i, j] = float(row[j])
        D = 0.5 * (D + D.T)

        X = np.full((N, N), 1.0 / N, dtype=float)
        K_iters = 30
        tol = 1e-9

        for k in range(K_iters):
            G = W @ X @ D + W.T @ X @ D.T
            row_ind, col_ind = linear_sum_assignment(G)
            S = np.zeros((N, N), dtype=float)
            S[row_ind, col_ind] = 1.0

            gamma = 2.0 / (k + 2.0)
            X_new = (1.0 - gamma) * X + gamma * S

            if np.linalg.norm(X_new - X) < tol:
                X = X_new
                break
            X = X_new

        row_ind, col_ind = linear_sum_assignment(-X)

        mapping = list(range(N))
        used = [False] * N
        for r, c in zip(row_ind, col_ind):
            mapping[r] = int(c)
            used[int(c)] = True

        if len(set(mapping)) != N:
            free = [p for p in range(N) if not used[p]]
            seen = set()
            fi = 0
            for r in range(N):
                if mapping[r] in seen:
                    mapping[r] = free[fi]
                    fi += 1
                seen.add(mapping[r])

        self.mapping_dict = mapping
        self.reverse_mapping_dict = [0] * N
        for logical, physical in enumerate(self.mapping_dict):
            self.reverse_mapping_dict[physical] = logical

    except Exception:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)