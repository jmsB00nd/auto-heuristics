def init_mapping(self):
    import numpy as np
    from scipy.linalg import orthogonal_procrustes, eigh
    from collections import defaultdict

    N = self.num_qubits

    # --- Step 1: Hardware coupling graph Laplacian & heat kernel embedding ---
    adj_hw = np.zeros((N, N), dtype=np.float64)
    for u in self.backend:
        for v in self.backend[u]:
            if u < N and v < N:
                adj_hw[u, v] = 1.0
                adj_hw[v, u] = 1.0

    deg_hw = np.diag(adj_hw.sum(axis=1))
    L_hw = deg_hw - adj_hw
    eigvals_hw, eigvecs_hw = eigh(L_hw)
    eigvals_hw = np.maximum(eigvals_hw, 0.0)

    time_scales = [0.1, 1.0, 10.0]
    d = len(time_scales)
    phys_embed = np.zeros((N, d), dtype=np.float64)
    for k, t in enumerate(time_scales):
        kernel_diag = np.exp(-eigvals_hw * t)
        coords = eigvecs_hw * kernel_diag[np.newaxis, :]
        phys_embed[:, k] = np.linalg.norm(coords, axis=1)

    # --- Step 2: Interaction graph from access2q ---
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(int)

    if self.access2q is not None:
        gate_source = self.access2q
    else:
        gate_source = {g: qs for g, qs in self.access.items() if len(qs) == 2}

    for gate, qubits in gate_source.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            edge = (min(q1, q2), max(q1, q2))
            interaction_weight[edge] += 1.0
            logical_degree[q1] += 1
            logical_degree[q2] += 1

    logical_qubits_with_interactions = set()
    for (q1, q2) in interaction_weight:
        logical_qubits_with_interactions.add(q1)
        logical_qubits_with_interactions.add(q2)

    adj_lg = np.zeros((N, N), dtype=np.float64)
    for (q1, q2), w in interaction_weight.items():
        if q1 < N and q2 < N:
            adj_lg[q1, q2] = w
            adj_lg[q2, q1] = w

    deg_lg = np.diag(adj_lg.sum(axis=1))
    L_lg = deg_lg - adj_lg
    eigvals_lg, eigvecs_lg = eigh(L_lg)
    eigvals_lg = np.maximum(eigvals_lg, 0.0)

    log_embed = np.zeros((N, d), dtype=np.float64)
    for k, t in enumerate(time_scales):
        kernel_diag = np.exp(-eigvals_lg * t)
        coords = eigvecs_lg * kernel_diag[np.newaxis, :]
        log_embed[:, k] = np.linalg.norm(coords, axis=1)

    # --- Step 3: Orthogonal Procrustes alignment ---
    log_center = log_embed - log_embed.mean(axis=0)
    phys_center = phys_embed - phys_embed.mean(axis=0)

    log_scale = np.linalg.norm(log_center)
    phys_scale = np.linalg.norm(phys_center)
    if log_scale > 1e-12:
        log_center /= log_scale
    if phys_scale > 1e-12:
        phys_center /= phys_scale

    R, _ = orthogonal_procrustes(log_center, phys_center)
    log_aligned = log_center @ R
    if phys_scale > 1e-12:
        log_aligned *= phys_scale

    phys_coords = phys_center
    if phys_scale > 1e-12:
        phys_coords = phys_center * phys_scale

    # --- Step 4 & 5: Greedy nearest-neighbor assignment by degree ---
    sorted_logical = sorted(logical_qubits_with_interactions,
                            key=lambda q: logical_degree.get(q, 0),
                            reverse=True)

    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))
    used_physical = set()
    mapped_logical = set()

    for lq in sorted_logical:
        lq_coord = log_aligned[lq]
        best_pq = None
        best_dist = float('inf')
        for pq in range(N):
            if pq in used_physical:
                continue
            dist = np.linalg.norm(lq_coord - phys_coords[pq])
            if dist < best_dist:
                best_dist = dist
                best_pq = pq
        if best_pq is not None:
            self.mapping_dict[lq] = best_pq
            self.reverse_mapping_dict[best_pq] = lq
            used_physical.add(best_pq)
            mapped_logical.add(lq)

    # --- Step 6: Fill remaining with identity-style assignment ---
    remaining_logical = [q for q in range(N) if q not in mapped_logical]
    remaining_physical = [q for q in range(N) if q not in used_physical]
    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)