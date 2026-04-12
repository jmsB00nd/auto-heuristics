def init_mapping(self):
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    num_q = self.num_qubits

    # Collect logical qubits used in the circuit
    logical_qubit_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())
    n = len(physical_qubits)

    # Build symmetric interaction weight matrix W (n x n)
    W = np.zeros((n, n), dtype=np.float64)
    # Map qubit ids to matrix indices
    lq_to_idx = {q: i for i, q in enumerate(physical_qubits)}  # use physical range as indices
    # For logical qubits, map them into the index space
    lq_idx = {}
    for i, q in enumerate(logical_qubits):
        lq_idx[q] = i  # logical qubits get indices 0..len(logical_qubits)-1

    n_logical = len(logical_qubits)
    W_logical = np.zeros((n, n), dtype=np.float64)
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            if q1 in lq_idx and q2 in lq_idx:
                i, j = lq_idx[q1], lq_idx[q2]
                W_logical[i][j] += 1.0
                W_logical[j][i] += 1.0

    # Build distance matrix D (n x n) using physical qubit ordering
    D = np.zeros((n, n), dtype=np.float64)
    for i, p1 in enumerate(physical_qubits):
        for j, p2 in enumerate(physical_qubits):
            D[i][j] = self.distance_matrix[p1][p2]

    # Frank-Wolfe algorithm on doubly-stochastic relaxation
    # Objective: minimize trace(W * P * D * P^T)
    # Start with identity as initial doubly-stochastic matrix
    P = np.eye(n, dtype=np.float64)

    max_iterations = 20
    for iteration in range(max_iterations):
        # Gradient of trace(W * P * D * P^T) w.r.t. P is 2 * W * P * D
        # (since W and D are symmetric)
        grad = 2.0 * W_logical @ P @ D

        # Solve linear assignment to find the permutation matrix Q
        # that minimizes trace(grad^T * Q) = sum_ij grad_ij * Q_ij
        row_ind, col_ind = linear_sum_assignment(grad)
        Q = np.zeros((n, n), dtype=np.float64)
        Q[row_ind, col_ind] = 1.0

        # Step size: use 2/(iteration+2) (standard Frank-Wolfe)
        gamma = 2.0 / (iteration + 2)

        # Update P as convex combination
        P_new = (1.0 - gamma) * P + gamma * Q

        P = P_new

    # Project final doubly-stochastic P to nearest permutation via Hungarian
    # Minimize -trace(P^T * Perm), equivalent to maximizing overlap
    cost = -P
    row_ind, col_ind = linear_sum_assignment(cost)

    # col_ind[i] = physical index that logical index i maps to
    # Build the mapping
    mapping_dict = list(range(num_q))
    reverse_mapping_dict = list(range(num_q))

    # Apply the computed assignment for logical qubits
    for i, lq in enumerate(logical_qubits):
        target_phys = physical_qubits[col_ind[i]]
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        # Swap to maintain bijection
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)