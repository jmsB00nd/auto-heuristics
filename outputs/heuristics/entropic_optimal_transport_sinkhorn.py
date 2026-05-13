def init_mapping(self):
    import numpy as np
    import math

    N = self.num_qubits

    # --- Collect 2q interactions and active logicals ---
    interactions = []
    active_logicals = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            interactions.append((q1, q2))
            active_logicals.add(q1)
            active_logicals.add(q2)

    # Logicals to consider — pad to N for square transport
    logicals_sorted = sorted(active_logicals)
    L_active = len(logicals_sorted)
    # Pad with placeholder logical ids (idle) up to N
    all_logicals = list(logicals_sorted)
    idle_pool = [q for q in range(N) if q not in active_logicals]
    for q in idle_pool:
        all_logicals.append(q)
    # Truncate or pad
    if len(all_logicals) < N:
        # add dummies (use -1 sentinels we will resolve later)
        while len(all_logicals) < N:
            all_logicals.append(-1)
    elif len(all_logicals) > N:
        all_logicals = all_logicals[:N]

    # --- Source mass mu over logicals (proportional to row activity) ---
    mu = np.zeros(N, dtype=np.float64)
    for i, lq in enumerate(all_logicals):
        if lq == -1:
            mu[i] = 1e-6
        else:
            act = float(self.logical_activity.get(lq, 0)) if hasattr(self.logical_activity, "get") else float(self.logical_activity[lq])
            mu[i] = act if act > 0 else 1e-6
    mu = mu / mu.sum()

    # --- Target mass nu uniform over physical qubits ---
    nu = np.full(N, 1.0 / N, dtype=np.float64)

    # --- Build QIG neighborhood vectors per logical (over all logicals) ---
    # Use a normalized neighborhood vector for divergence comparison
    logical_index = {lq: i for i, lq in enumerate(all_logicals) if lq != -1}
    L_neigh = np.zeros((N, N), dtype=np.float64)
    for i, lq in enumerate(all_logicals):
        if lq == -1:
            continue
        neighbors = self.qubit_interaction_graph.get(lq, {}) if hasattr(self.qubit_interaction_graph, "get") else self.qubit_interaction_graph[lq]
        for nb, w in neighbors.items():
            if nb in logical_index:
                L_neigh[i, logical_index[nb]] = float(w)
    # Row-normalize to distributions for divergence
    row_sums = L_neigh.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    L_dist = L_neigh / row_sums

    # --- Build physical neighborhood vectors using distance/adjacency ---
    P_neigh = np.zeros((N, N), dtype=np.float64)
    for p in range(N):
        for q in range(N):
            if p == q:
                continue
            d = self.distance_matrix[p][q]
            if d > 0:
                P_neigh[p, q] = 1.0 / d
    p_row_sums = P_neigh.sum(axis=1, keepdims=True)
    p_row_sums[p_row_sums == 0] = 1.0
    P_dist = P_neigh / p_row_sums

    # --- Physical eccentricity (lower centrality => higher eccentricity) ---
    ecc = np.zeros(N, dtype=np.float64)
    cmax = 0.0
    for p in range(N):
        c = float(self.physical_centrality.get(p, 0.0)) if hasattr(self.physical_centrality, "get") else 0.0
        if c > cmax:
            cmax = c
    if cmax <= 0:
        cmax = 1.0
    for p in range(N):
        c = float(self.physical_centrality.get(p, 0.0)) if hasattr(self.physical_centrality, "get") else 0.0
        ecc[p] = 1.0 - (c / cmax)

    # --- Ground cost C[L,P]: divergence between neighborhood distributions + eccentricity weighted by activity ---
    # Use symmetric KL-like divergence between L_dist[i] and P_dist[p]
    eps_safe = 1e-12
    C = np.zeros((N, N), dtype=np.float64)
    log_L = np.log(L_dist + eps_safe)
    log_P = np.log(P_dist + eps_safe)
    # KL(L||P) = sum L * (log L - log P)
    for i in range(N):
        kl_lp = (L_dist[i] * (log_L[i] - log_P)).sum(axis=1)
        kl_pl = (P_dist * (log_P - log_L[i])).sum(axis=1)
        C[i, :] = 0.5 * (kl_lp + kl_pl)
    # Add eccentricity term scaled by source mass (activity)
    activity_scale = mu / (mu.max() + eps_safe)
    C = C + activity_scale[:, None] * ecc[None, :]

    # Normalize cost for numerical stability
    cmax_c = C.max()
    if cmax_c > 0:
        C = C / cmax_c

    # --- Sinkhorn-Knopp ---
    reg = 0.05
    K = np.exp(-C / reg)
    # Avoid zeros
    K = np.maximum(K, 1e-300)
    u = np.ones(N, dtype=np.float64)
    v = np.ones(N, dtype=np.float64)
    max_iter = 200
    tol = 1e-7
    for _ in range(max_iter):
        u_new = mu / (K @ v + 1e-300)
        v_new = nu / (K.T @ u_new + 1e-300)
        if np.max(np.abs(u_new - u)) < tol and np.max(np.abs(v_new - v)) < tol:
            u, v = u_new, v_new
            break
        u, v = u_new, v_new
    T = (u[:, None] * K) * v[None, :]

    # --- Greedy max-marginal extraction ---
    mapping = [-1] * N
    reverse = [-1] * N
    used_phys = [False] * N
    used_log = [False] * N

    # Sort all (i,p) by T value desc; for memory cap to top entries
    flat_idx = np.argsort(-T, axis=None)
    assigned = 0
    for idx in flat_idx:
        if assigned >= N:
            break
        i = idx // N
        p = idx % N
        if used_log[i] or used_phys[p]:
            continue
        lq = all_logicals[i]
        if lq == -1:
            used_log[i] = True
            continue
        mapping[lq] = p
        reverse[p] = lq
        used_log[i] = True
        used_phys[p] = True
        assigned += 1

    # --- Identity fallback for any logical not yet assigned ---
    remaining_phys = [p for p in range(N) if not used_phys[p]]
    rp_idx = 0
    for lq in range(N):
        if mapping[lq] == -1:
            # find next free physical
            while rp_idx < len(remaining_phys) and used_phys[remaining_phys[rp_idx]]:
                rp_idx += 1
            if rp_idx < len(remaining_phys):
                p = remaining_phys[rp_idx]
                mapping[lq] = p
                reverse[p] = lq
                used_phys[p] = True
                rp_idx += 1
            else:
                # last-ditch: scan
                for p in range(N):
                    if not used_phys[p]:
                        mapping[lq] = p
                        reverse[p] = lq
                        used_phys[p] = True
                        break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)