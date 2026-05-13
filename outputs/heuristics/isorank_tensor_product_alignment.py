def init_mapping(self):
    import numpy as np

    N = self.num_qubits

    # Identify logical qubits actually used by the circuit
    logical_qubits = set()
    for _, qubits in self.access.items():
        for q in qubits:
            if 0 <= q < N:
                logical_qubits.add(q)

    # --- Logical adjacency A_L (weighted, symmetric) ---
    A_L = np.zeros((N, N), dtype=float)
    qig = self.qubit_interaction_graph
    if qig:
        for q1 in qig:
            if q1 >= N:
                continue
            for q2, w in qig[q1].items():
                if 0 <= q2 < N:
                    A_L[q1, q2] = float(w)
    else:
        for _, qubits in self.access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                if 0 <= a < N and 0 <= b < N and a != b:
                    A_L[a, b] += 1.0
                    A_L[b, a] += 1.0
    A_L = np.maximum(A_L, A_L.T)

    # --- Physical adjacency A_P (unweighted, symmetric) ---
    A_P = np.zeros((N, N), dtype=float)
    for p1 in self.backend:
        if p1 >= N:
            continue
        for p2 in self.backend[p1]:
            if 0 <= p2 < N:
                A_P[p1, p2] = 1.0
                A_P[p2, p1] = 1.0

    def row_stochastic(M):
        s = M.sum(axis=1, keepdims=True)
        out = np.where(s > 0, M / np.maximum(s, 1e-12), 1.0 / N)
        return out

    W_L = row_stochastic(A_L)
    W_P = row_stochastic(A_P)

    # --- Personalization prior H = activity ⊗ centrality ---
    activity = np.zeros(N, dtype=float)
    for q, a in self.logical_activity.items():
        if 0 <= q < N:
            activity[q] = float(a)
    if activity.sum() <= 0:
        activity = np.ones(N, dtype=float)
    activity = activity / activity.sum()

    centrality = np.zeros(N, dtype=float)
    for p, c in self.physical_centrality.items():
        if 0 <= p < N:
            centrality[p] = float(c)
    if centrality.sum() <= 0:
        centrality = np.ones(N, dtype=float)
    centrality = centrality / centrality.sum()

    H = np.outer(activity, centrality)
    Hs = H.sum()
    if Hs > 0:
        H = H / Hs
    else:
        H = np.full((N, N), 1.0 / (N * N))

    # --- IsoRank power iteration on the Kronecker product ---
    alpha = 0.85
    R = H.copy()
    for _ in range(30):
        R_new = alpha * (W_L.T @ R @ W_P) + (1.0 - alpha) * H
        s = R_new.sum()
        if s > 0:
            R_new /= s
        if np.abs(R_new - R).sum() < 1e-9:
            R = R_new
            break
        R = R_new

    # --- Greedy decode: highest similarity first ---
    mapping_dict = [-1] * N
    reverse_mapping_dict = [-1] * N
    used_logical = set()
    used_physical = set()

    flat_order = np.argsort(-R, axis=None, kind='stable')
    target_count = len(logical_qubits)
    committed = 0
    for idx in flat_order:
        L = int(idx // N)
        P = int(idx % N)
        if L not in logical_qubits:
            continue
        if L in used_logical or P in used_physical:
            continue
        mapping_dict[L] = P
        reverse_mapping_dict[P] = L
        used_logical.add(L)
        used_physical.add(P)
        committed += 1
        if committed >= target_count:
            break

    # --- Back-fill remaining logicals onto remaining physicals ---
    remaining_physicals = [p for p in range(N) if p not in used_physical]
    rp_idx = 0
    for L in range(N):
        if mapping_dict[L] == -1:
            while rp_idx < len(remaining_physicals) and remaining_physicals[rp_idx] in used_physical:
                rp_idx += 1
            if rp_idx < len(remaining_physicals):
                p = remaining_physicals[rp_idx]
                rp_idx += 1
                mapping_dict[L] = p
                reverse_mapping_dict[p] = L
                used_physical.add(p)

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)