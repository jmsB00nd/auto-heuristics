def init_mapping(self):
    import numpy as np

    N = self.num_qubits

    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            logical_qubits.add(qubits[0])
            logical_qubits.add(qubits[1])
    logical_qubits = sorted(q for q in logical_qubits if 0 <= q < N)

    if len(logical_qubits) == 0:
        for q in range(N):
            self.mapping_dict[q] = q
            self.reverse_mapping_dict[q] = q
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    L_log = np.zeros((N, N), dtype=float)
    for q1, neigh in self.qubit_interaction_graph.items():
        if not (0 <= q1 < N):
            continue
        for q2, w in neigh.items():
            if 0 <= q2 < N and q1 != q2:
                L_log[q1, q2] = float(w)
    L_log = 0.5 * (L_log + L_log.T)

    L_phys = np.zeros((N, N), dtype=float)
    for (a, b) in self.backend_connections:
        if 0 <= a < N and 0 <= b < N and a != b:
            L_phys[a, b] = 1.0
            L_phys[b, a] = 1.0

    def laplacian_embedding(A, k):
        n = A.shape[0]
        deg = A.sum(axis=1)
        D = np.diag(deg)
        L = D - A
        try:
            vals, vecs = np.linalg.eigh(L)
        except Exception:
            return np.zeros((n, k))
        order = np.argsort(vals)
        vecs = vecs[:, order]
        start = 1 if n > k + 1 else 0
        end = min(start + k, n)
        emb = vecs[:, start:end]
        if emb.shape[1] < k:
            pad = np.zeros((n, k - emb.shape[1]))
            emb = np.hstack([emb, pad])
        return emb

    k = min(3, max(2, N - 1))
    X_full = laplacian_embedding(L_log, k)
    Y_full = laplacian_embedding(L_phys, k)

    X_log = X_full[logical_qubits, :]
    physical_indices = list(range(N))
    Y_phys = Y_full[physical_indices, :]

    try:
        M = X_log.T @ Y_phys[:len(logical_qubits), :] if len(logical_qubits) <= N else X_log.T @ Y_phys
        if len(logical_qubits) >= 1:
            m = min(len(logical_qubits), N)
            M = X_log.T @ Y_phys[:m, :]
            U, _, Vt = np.linalg.svd(M, full_matrices=False)
            R = U @ Vt
            X_rot = X_log @ R
        else:
            X_rot = X_log
    except Exception:
        X_rot = X_log

    activity = self.logical_activity if self.logical_activity else {}
    order_logicals = sorted(logical_qubits, key=lambda q: -float(activity.get(q, 0)))

    used_physical = set()
    logical_to_idx = {q: i for i, q in enumerate(logical_qubits)}

    for lq in order_logicals:
        idx = logical_to_idx[lq]
        point = X_rot[idx]
        best_p = None
        best_d = float('inf')
        for p in range(N):
            if p in used_physical:
                continue
            diff = Y_phys[p] - point
            d = float(np.dot(diff, diff))
            if d < best_d:
                best_d = d
                best_p = p
        if best_p is None:
            for p in range(N):
                if p not in used_physical:
                    best_p = p
                    break
        self.mapping_dict[lq] = best_p
        self.reverse_mapping_dict[best_p] = lq
        used_physical.add(best_p)

    remaining_physical = [p for p in range(N) if p not in used_physical]
    rp_iter = iter(remaining_physical)
    for q in range(N):
        if self.mapping_dict[q] is None:
            try:
                p = next(rp_iter)
            except StopIteration:
                for pp in range(N):
                    if pp not in used_physical:
                        p = pp
                        break
            self.mapping_dict[q] = p
            self.reverse_mapping_dict[p] = q
            used_physical.add(p)

    for p in range(N):
        if self.reverse_mapping_dict[p] is None:
            for q in range(N):
                if self.mapping_dict[q] == p:
                    self.reverse_mapping_dict[p] = q
                    break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)