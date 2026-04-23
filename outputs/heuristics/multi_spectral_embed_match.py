def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from scipy.linalg import orthogonal_procrustes

    n = self.num_qubits

    # Collect logical qubits that participate in 2-qubit gates
    logical_qubits = sorted(self.qubit_interaction_graph.keys())
    num_logical = len(logical_qubits)

    if num_logical <= 1:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    # Physical qubits from backend
    physical_qubits = sorted(self.backend.keys())
    num_physical = len(physical_qubits)

    # --- Build logical interaction graph Laplacian ---
    log_idx = {q: i for i, q in enumerate(logical_qubits)}
    L_log = np.zeros((num_logical, num_logical))
    for q1 in logical_qubits:
        for q2, w in self.qubit_interaction_graph[q1].items():
            if q2 in log_idx:
                i, j = log_idx[q1], log_idx[q2]
                L_log[i, j] = -w
                L_log[i, i] += w

    # --- Build physical coupling graph Laplacian ---
    phys_idx = {q: i for i, q in enumerate(physical_qubits)}
    L_phys = np.zeros((num_physical, num_physical))
    for q1 in physical_qubits:
        for q2 in self.backend[q1]:
            if q2 in phys_idx:
                i, j = phys_idx[q1], phys_idx[q2]
                L_phys[i, j] = -1
                L_phys[i, i] += 1

    # --- Spectral embedding ---
    k = min(num_logical, num_physical, 8)
    k = max(k, 2)

    eigvals_log, eigvecs_log = np.linalg.eigh(L_log)
    # Skip the trivial zero eigenvalue (index 0), take next k
    end_log = min(1 + k, num_logical)
    embed_log = eigvecs_log[:, 1:end_log]

    eigvals_phys, eigvecs_phys = np.linalg.eigh(L_phys)
    end_phys = min(1 + k, num_physical)
    embed_phys = eigvecs_phys[:, 1:end_phys]

    # Pad to same number of columns if needed
    dim = max(embed_log.shape[1], embed_phys.shape[1])
    if embed_log.shape[1] < dim:
        embed_log = np.hstack([embed_log, np.zeros((embed_log.shape[0], dim - embed_log.shape[1]))])
    if embed_phys.shape[1] < dim:
        embed_phys = np.hstack([embed_phys, np.zeros((embed_phys.shape[0], dim - embed_phys.shape[1]))])

    # --- Align embeddings via Procrustes ---
    # Pick anchor points: top-k logical by activity, match to physical by centrality
    # Use a simple approach: align using Procrustes on the full sets (padded to same size)
    m = max(num_logical, num_physical)
    padded_log = np.zeros((m, dim))
    padded_phys = np.zeros((m, dim))
    padded_log[:num_logical] = embed_log
    padded_phys[:num_physical] = embed_phys

    R, _ = orthogonal_procrustes(padded_log, padded_phys)
    aligned_log = embed_log @ R

    # --- Build cost matrix and solve assignment ---
    cost = np.zeros((num_logical, num_physical))
    for i in range(num_logical):
        for j in range(num_physical):
            cost[i, j] = np.linalg.norm(aligned_log[i] - embed_phys[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    # --- Populate mapping ---
    self.mapping_dict = list(range(n))
    self.reverse_mapping_dict = list(range(n))

    used_physical = set()
    assigned_logical = set()

    for r, c in zip(row_ind, col_ind):
        lq = logical_qubits[r]
        pq = physical_qubits[c]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)
        assigned_logical.add(lq)

    # Fill remaining logical qubits with unused physical qubits
    unused_physical = [q for q in range(n) if q not in used_physical]
    unassigned_logical = [q for q in range(n) if q not in assigned_logical]

    for lq, pq in zip(unassigned_logical, unused_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)