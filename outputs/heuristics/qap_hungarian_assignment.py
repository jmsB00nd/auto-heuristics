def init_mapping(self):
    N = self.num_qubits

    def _identity_fallback():
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        D = self.distance_matrix
        if isinstance(D, list):
            D_arr = np.array(D, dtype=float)
        else:
            D_arr = np.asarray(D, dtype=float)
        if D_arr.shape[0] != N or D_arr.shape[1] != N:
            _identity_fallback()
            assert len(set(self.mapping_dict)) == len(self.mapping_dict)
            return

        W = np.zeros((N, N), dtype=float)
        active_logicals = set()
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                if 0 <= a < N and 0 <= b < N and a != b:
                    W[a][b] += 1.0
                    W[b][a] += 1.0
                    active_logicals.add(a)
                    active_logicals.add(b)

        pi_hat = np.arange(N)
        try:
            centrality = self.physical_centrality
            if centrality:
                activity = self.logical_activity
                logicals_sorted = sorted(range(N), key=lambda q: -activity.get(q, 0))
                physicals_sorted = sorted(range(N), key=lambda p: -centrality.get(p, 0.0))
                seeded = np.arange(N)
                for L, P in zip(logicals_sorted, physicals_sorted):
                    seeded[L] = P
                if len(set(seeded.tolist())) == N:
                    pi_hat = seeded
        except Exception:
            pi_hat = np.arange(N)

        max_iter = 8
        best_cost = float("inf")
        best_pi = pi_hat.copy()
        for _ in range(max_iter):
            D_pi = D_arr[:, pi_hat]
            C = W @ D_pi
            row_ind, col_ind = linear_sum_assignment(C)
            new_pi = np.empty(N, dtype=int)
            new_pi[row_ind] = col_ind
            cost = float(C[row_ind, col_ind].sum())
            if cost < best_cost - 1e-12:
                best_cost = cost
                best_pi = new_pi.copy()
            if np.array_equal(new_pi, pi_hat):
                pi_hat = new_pi
                break
            pi_hat = new_pi

        chosen = best_pi if best_cost < float("inf") else pi_hat
        mapping = chosen.tolist()
        if len(set(mapping)) != N:
            _identity_fallback()
            assert len(set(self.mapping_dict)) == len(self.mapping_dict)
            return

        self.mapping_dict = [int(x) for x in mapping]
        self.reverse_mapping_dict = [0] * N
        for L in range(N):
            self.reverse_mapping_dict[self.mapping_dict[L]] = L
    except Exception:
        _identity_fallback()

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)