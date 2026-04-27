def init_mapping(self):
    import numpy as np
    N = self.num_qubits

    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    try:
        import networkx as nx
        from scipy.optimize import linear_sum_assignment
        from scipy.linalg import eigh

        interactions = []
        active = set()
        for gate_id, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = int(qubits[0]), int(qubits[1])
                if a == b:
                    continue
                interactions.append((a, b))
                active.add(a)
                active.add(b)

        max_logical = max(active) + 1 if active else 0
        n_logical_nodes = max(max_logical, N)
        if n_logical_nodes > N:
            return

        W_log = np.zeros((N, N), dtype=float)
        for a, b in interactions:
            if a < N and b < N:
                W_log[a, b] += 1.0
                W_log[b, a] += 1.0

        W_phy = np.zeros((N, N), dtype=float)
        for (u, v) in self.backend_connections:
            if 0 <= u < N and 0 <= v < N and u != v:
                W_phy[u, v] = 1.0
                W_phy[v, u] = 1.0

        def normalized_laplacian(W):
            d = W.sum(axis=1)
            with np.errstate(divide='ignore'):
                d_inv_sqrt = np.where(d > 0, 1.0 / np.sqrt(d), 0.0)
            D_inv_sqrt = np.diag(d_inv_sqrt)
            L = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt
            L = 0.5 * (L + L.T)
            return L

        L_log = normalized_laplacian(W_log)
        L_phy = normalized_laplacian(W_phy)

        k = min(max(2, int(np.ceil(np.log2(max(N, 2)))) + 1), N - 1)
        if k < 1:
            k = 1

        w_log, V_log = eigh(L_log)
        w_phy, V_phy = eigh(L_phy)

        emb_log = V_log[:, 1:1 + k]
        emb_phy = V_phy[:, 1:1 + k]

        def normalize_cols(M):
            norms = np.linalg.norm(M, axis=0)
            norms = np.where(norms > 1e-12, norms, 1.0)
            return M / norms

        emb_log = normalize_cols(emb_log)
        emb_phy = normalize_cols(emb_phy)

        for j in range(emb_log.shape[1]):
            d_pos = np.linalg.norm(emb_log[:, j] - emb_phy[:, j])
            d_neg = np.linalg.norm(emb_log[:, j] + emb_phy[:, j])
            if d_neg < d_pos:
                emb_log[:, j] = -emb_log[:, j]

        diff = emb_log[:, None, :] - emb_phy[None, :, :]
        cost = np.sum(diff * diff, axis=2)

        row_ind, col_ind = linear_sum_assignment(cost)

        new_map = [0] * N
        new_rev = [0] * N
        for L_idx, P_idx in zip(row_ind, col_ind):
            new_map[int(L_idx)] = int(P_idx)
            new_rev[int(P_idx)] = int(L_idx)

        self.mapping_dict = new_map
        self.reverse_mapping_dict = new_rev
    except Exception:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)