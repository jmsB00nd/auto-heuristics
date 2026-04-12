def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    from collections import defaultdict

    n = self.num_qubits

    # --- Step 1: Identify logical qubits used in 2-qubit gates ---
    interaction_weights = defaultdict(float)
    logical_qubits_used = set()
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_used.add(q1)
            logical_qubits_used.add(q2)
            interaction_weights[(q1, q2)] += 1.0
            interaction_weights[(q2, q1)] += 1.0

    num_logical = len(logical_qubits_used)

    # Fallback: if no 2-qubit gates, use identity mapping
    if num_logical < 2:
        self.mapping_dict = list(range(n))
        self.reverse_mapping_dict = list(range(n))
        if self.use_isl:
            from src.utils.python_to_isl import dict_to_isl_map
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    logical_list = sorted(logical_qubits_used)
    log_idx = {q: i for i, q in enumerate(logical_list)}

    # --- Step 2: Build weighted Laplacian of logical interaction graph ---
    L_log = np.zeros((num_logical, num_logical))
    for (q1, q2), w in interaction_weights.items():
        if q1 < q2:
            i, j = log_idx[q1], log_idx[q2]
            L_log[i, j] = -w
            L_log[j, i] = -w
            L_log[i, i] += w
            L_log[j, j] += w

    # --- Step 3: Build Laplacian of hardware coupling graph ---
    phys_list = sorted(self.backend.keys())
    num_phys = len(phys_list)
    phys_idx = {q: i for i, q in enumerate(phys_list)}

    L_hw = np.zeros((num_phys, num_phys))
    seen_edges = set()
    for p in phys_list:
        for nb in self.backend[p]:
            if nb in phys_idx and (p, nb) not in seen_edges:
                seen_edges.add((p, nb))
                seen_edges.add((nb, p))
                i, j = phys_idx[p], phys_idx[nb]
                L_hw[i, j] = -1.0
                L_hw[j, i] = -1.0
                L_hw[i, i] += 1.0
                L_hw[j, j] += 1.0

    # --- Step 4: Compute spectral embeddings (top-k smallest non-trivial eigenvectors) ---
    k = min(5, num_logical - 1, num_phys - 1)
    if k < 1:
        k = 1

    eigvals_log, eigvecs_log = np.linalg.eigh(L_log)
    eigvals_hw, eigvecs_hw = np.linalg.eigh(L_hw)

    # Skip the first eigenvector (constant/Fiedler = index 0 for eigh which sorts ascending)
    # Take eigenvectors 1..k (the smallest non-trivial ones)
    embed_log = eigvecs_log[:, 1:1+k]  # shape: (num_logical, k)
    embed_hw = eigvecs_hw[:, 1:1+k]    # shape: (num_phys, k)

    # --- Step 5: Procrustes alignment ---
    # We want to find orthogonal R that minimizes ||embed_log - embed_hw_subset @ R||
    # But sizes differ, so we align the logical embedding to the physical embedding space.
    # Use the centroids of both for translation, then SVD for rotation.

    centroid_log = embed_log.mean(axis=0)
    centroid_hw = embed_hw.mean(axis=0)

    embed_log_centered = embed_log - centroid_log
    embed_hw_centered = embed_hw - centroid_hw

    # Build cost matrix: for each (logical_i, physical_j), compute distance after alignment
    # Since we can't do Procrustes without correspondence, we use a simpler approach:
    # Normalize both embeddings, then compute pairwise distances.
    # Better: do Procrustes using the k dimensions directly.

    # Normalize scale of both embeddings (Frobenius norm)
    norm_log = np.linalg.norm(embed_log_centered, 'fro')
    norm_hw = np.linalg.norm(embed_hw_centered, 'fro')

    if norm_log > 1e-10:
        embed_log_centered /= norm_log
    if norm_hw > 1e-10:
        embed_hw_centered /= norm_hw

    # Procrustes: find R minimizing ||L - P @ R|| over orthogonal R
    # M = L^T @ P, then SVD(M) = U S V^T, R = V @ U^T
    # But L is (num_logical, k) and P is (num_phys, k) - different row counts.
    # We compute cost matrix directly: C[i,j] = ||L_i - R * P_j|| 
    # For proper alignment, we use an iterative or approximate approach.
    # 
    # Alternative: Since both are in the same eigenvector space (just different graphs),
    # we can directly compute pairwise Euclidean distances after normalization.
    # The sign ambiguity of eigenvectors is the main issue - we resolve by trying both signs.

    # Handle eigenvector sign ambiguity: for each dimension, pick sign that maximizes
    # correlation between the two sets of embeddings (using degree as proxy)
    log_degrees = np.diag(L_log)
    hw_degrees = np.diag(L_hw)

    for d in range(k):
        # Correlation of eigenvector component with degree
        corr_log = np.corrcoef(embed_log_centered[:, d], log_degrees)[0, 1] if num_logical > 1 else 0
        corr_hw = np.corrcoef(embed_hw_centered[:, d], hw_degrees)[0, 1] if num_phys > 1 else 0
        # Flip hardware eigenvector sign if correlations disagree
        if corr_log * corr_hw < 0:
            embed_hw_centered[:, d] *= -1

    # --- Step 6: Build cost matrix and solve assignment ---
    # C[i, j] = ||embed_log_centered[i] - embed_hw_centered[j]||_2
    # Using broadcasting: (num_logical, 1, k) - (1, num_phys, k)
    diff = embed_log_centered[:, np.newaxis, :] - embed_hw_centered[np.newaxis, :, :]
    cost_matrix = np.linalg.norm(diff, axis=2)  # (num_logical, num_phys)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # --- Step 7: Build the mapping ---
    # Map logical qubit logical_list[row_ind[i]] -> physical qubit phys_list[col_ind[i]]
    lq_to_phys = {}
    for r, c in zip(row_ind, col_ind):
        lq_to_phys[logical_list[r]] = phys_list[c]

    # Start with identity and swap into place to maintain bijectivity
    mapping_dict = list(range(n))
    reverse_mapping_dict = list(range(n))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        from src.utils.python_to_isl import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)