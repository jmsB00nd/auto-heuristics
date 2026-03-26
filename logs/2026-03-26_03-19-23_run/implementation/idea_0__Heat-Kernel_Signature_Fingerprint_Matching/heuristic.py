def init_mapping(self):
    """
    Heat-Kernel Signature (HKS) Fingerprint Matching for Initial Mapping.

    Computes HKS at multiple time scales for nodes in both the logical
    interaction graph and the physical coupling graph. Matches logical to
    physical qubits by minimizing L2 distance between HKS vectors via
    the Hungarian algorithm, then refines with swap-based local search.
    """
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    gates_list = list(self.access.items())

    # Collect logical qubits used in the circuit
    logical_qubit_set = set()
    for _, qubits in gates_list:
        for q in qubits:
            logical_qubit_set.add(q)

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())
    n_logical = len(logical_qubits)
    n_physical = len(physical_qubits)

    # Trivial case: no logical qubits or single qubit
    if n_logical <= 1:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            from src.utils.isl_data_loader import dict_to_isl_map
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 1: Build interaction weights for logical graph ---
    interaction_weight = defaultdict(float)
    for _, qubits in gates_list:
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0

    # --- Step 2: Build weighted Laplacian for logical interaction graph ---
    lq_index = {q: i for i, q in enumerate(logical_qubits)}
    L_logical = np.zeros((n_logical, n_logical), dtype=np.float64)
    for (q1, q2), w in interaction_weight.items():
        i, j = lq_index[q1], lq_index[q2]
        L_logical[i, j] -= w
        L_logical[j, i] -= w
        L_logical[i, i] += w
        L_logical[j, j] += w

    # --- Step 3: Build unweighted Laplacian for hardware graph ---
    pq_index = {q: i for i, q in enumerate(physical_qubits)}
    L_hw = np.zeros((n_physical, n_physical), dtype=np.float64)
    seen_edges = set()
    for pq in physical_qubits:
        for neighbor in self.backend[pq]:
            if neighbor in pq_index:
                edge = (min(pq, neighbor), max(pq, neighbor))
                if edge not in seen_edges:
                    seen_edges.add(edge)
                    i, j = pq_index[pq], pq_index[neighbor]
                    L_hw[i, j] -= 1.0
                    L_hw[j, i] -= 1.0
                    L_hw[i, i] += 1.0
                    L_hw[j, j] += 1.0

    # --- Step 4: Eigendecomposition of both Laplacians ---
    eigenvalues_log, eigenvectors_log = np.linalg.eigh(L_logical)
    eigenvalues_hw, eigenvectors_hw = np.linalg.eigh(L_hw)

    # Clamp small negative eigenvalues to zero (numerical noise)
    eigenvalues_log = np.maximum(eigenvalues_log, 0.0)
    eigenvalues_hw = np.maximum(eigenvalues_hw, 0.0)

    # --- Step 5: Compute HKS at multiple time scales ---
    time_scales = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    n_times = len(time_scales)

    # HKS for logical graph: shape (n_logical, n_times)
    hks_logical = np.zeros((n_logical, n_times), dtype=np.float64)
    for t_idx, t in enumerate(time_scales):
        heat_kernel_diag = np.exp(-eigenvalues_log * t)
        phi_sq = eigenvectors_log ** 2
        hks_logical[:, t_idx] = phi_sq @ heat_kernel_diag

    # HKS for hardware graph: shape (n_physical, n_times)
    hks_hw = np.zeros((n_physical, n_times), dtype=np.float64)
    for t_idx, t in enumerate(time_scales):
        heat_kernel_diag = np.exp(-eigenvalues_hw * t)
        phi_sq = eigenvectors_hw ** 2
        hks_hw[:, t_idx] = phi_sq @ heat_kernel_diag

    # --- Step 6: Normalize HKS vectors for better matching ---
    for t_idx in range(n_times):
        col_log = hks_logical[:, t_idx]
        std_log = np.std(col_log)
        if std_log > 1e-12:
            hks_logical[:, t_idx] = (col_log - np.mean(col_log)) / std_log

        col_hw = hks_hw[:, t_idx]
        std_hw = np.std(col_hw)
        if std_hw > 1e-12:
            hks_hw[:, t_idx] = (col_hw - np.mean(col_hw)) / std_hw

    # --- Step 7: Build cost matrix C[i,j] = ||HKS_logical(i) - HKS_hw(j)||^2 ---
    diff = hks_logical[:, np.newaxis, :] - hks_hw[np.newaxis, :, :]
    cost_matrix = np.sum(diff ** 2, axis=2)

    # --- Step 8: Solve assignment via Hungarian algorithm ---
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    lq_to_phys = {}
    assigned_physical = set()
    for r, c in zip(row_ind, col_ind):
        lq = logical_qubits[r]
        pq = physical_qubits[c]
        lq_to_phys[lq] = pq
        assigned_physical.add(pq)

    # Assign remaining logical qubits to unassigned physical qubits greedily
    remaining_physical = [pq for pq in physical_qubits if pq not in assigned_physical]
    remaining_logical = [lq for lq in logical_qubits if lq not in lq_to_phys]
    for lq, pq in zip(remaining_logical, remaining_physical):
        lq_to_phys[lq] = pq
        assigned_physical.add(pq)

    # --- Step 9: Build full mapping arrays ---
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    # Apply assignments via in-place swaps to maintain bijection
    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        other_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[other_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = other_lq

    # --- Step 10: Local search refinement (swap-based steepest descent) ---
    def evaluate_cost(m_dict):
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            p1, p2 = m_dict[q1], m_dict[q2]
            cost += w * self.distance_matrix[p1][p2]
        return cost

    best_cost = evaluate_cost(mapping_dict)
    improved = True
    max_iterations = 50

    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        best_swap = None
        best_new_cost = best_cost

        for i in range(len(logical_qubits)):
            for j in range(i + 1, len(logical_qubits)):
                lq_i = logical_qubits[i]
                lq_j = logical_qubits[j]
                p_i = mapping_dict[lq_i]
                p_j = mapping_dict[lq_j]

                # Compute delta cost instead of full re-evaluation
                delta = 0.0
                for (q1, q2), w in interaction_weight.items():
                    involves_i = (q1 == lq_i or q2 == lq_i)
                    involves_j = (q1 == lq_j or q2 == lq_j)
                    if not involves_i and not involves_j:
                        continue
                    old_p1 = mapping_dict[q1]
                    old_p2 = mapping_dict[q2]
                    old_dist = self.distance_matrix[old_p1][old_p2]
                    new_p1 = p_j if q1 == lq_i else (p_i if q1 == lq_j else old_p1)
                    new_p2 = p_j if q2 == lq_i else (p_i if q2 == lq_j else old_p2)
                    new_dist = self.distance_matrix[new_p1][new_p2]
                    delta += w * (new_dist - old_dist)

                new_cost = best_cost + delta
                if new_cost < best_new_cost - 1e-10:
                    best_new_cost = new_cost
                    best_swap = (lq_i, lq_j)

        if best_swap is not None:
            lq_i, lq_j = best_swap
            p_i = mapping_dict[lq_i]
            p_j = mapping_dict[lq_j]
            mapping_dict[lq_i] = p_j
            mapping_dict[lq_j] = p_i
            reverse_mapping_dict[p_i] = lq_j
            reverse_mapping_dict[p_j] = lq_i
            best_cost = best_new_cost
            improved = True

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    if self.use_isl:
        from src.utils.isl_data_loader import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)