def init_mapping(self):
    import numpy as np
    from collections import defaultdict
    from scipy.optimize import linear_sum_assignment

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)
    phys_idx = {pq: i for i, pq in enumerate(physical_qubits)}

    # --- Step 1: Build interaction graph with front-layer bias ---
    logical_qubit_set = set()
    interaction_count = defaultdict(float)
    logical_neighbors = defaultdict(lambda: defaultdict(float))

    # Determine early gates for 2x front-layer bias
    gate_keys = sorted(self.access.keys())
    early_cutoff = len(gate_keys) // 4 if gate_keys else 0
    early_gates = set(gate_keys[:max(early_cutoff, 1)])

    for gate, qubits in self.access.items():
        if len(qubits) >= 2:
            q1, q2 = qubits[0], qubits[1]
            logical_qubit_set.add(q1)
            logical_qubit_set.add(q2)
            weight = 2.0 if gate in early_gates else 1.0
            key = (min(q1, q2), max(q1, q2))
            interaction_count[key] += weight
            logical_neighbors[q1][q2] += weight
            logical_neighbors[q2][q1] += weight
        elif len(qubits) == 1:
            logical_qubit_set.add(qubits[0])

    logical_qubits = sorted(logical_qubit_set)
    n_logical = len(logical_qubits)
    lq_idx = {lq: i for i, lq in enumerate(logical_qubits)}

    # --- Step 2: Compute degree centrality of physical qubits ---
    phys_degree = {}
    for pq in physical_qubits:
        phys_degree[pq] = len(self.backend.get(pq, []))
    max_degree = max(phys_degree.values()) if phys_degree else 1

    # Centrality score: higher is more central (inverse of avg distance)
    phys_centrality = {}
    for pq in physical_qubits:
        avg_dist = sum(self.distance_matrix[pq][other] for other in physical_qubits) / n_phys
        phys_centrality[pq] = 1.0 / (avg_dist + 1e-9)

    # Logical qubit total interaction weight (degree in interaction graph)
    logical_total_weight = defaultdict(float)
    for lq in logical_qubits:
        logical_total_weight[lq] = sum(logical_neighbors[lq].values())

    # --- Step 3: First pass - rough cost using centrality ---
    # High-interaction logical qubits prefer central physical qubits (low cost)
    # Cost = -centrality * interaction_degree (negative because we want to minimize)
    # But we need a proper cost matrix for assignment, so:
    # cost[i][j] = -logical_total_weight[lq_i] * phys_centrality[pq_j]
    cost_matrix_pass1 = np.zeros((n_logical, n_phys))
    for i, lq in enumerate(logical_qubits):
        w = logical_total_weight[lq]
        for j, pq in enumerate(physical_qubits):
            cost_matrix_pass1[i][j] = -w * phys_centrality[pq]

    # Solve first pass
    if n_logical > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix_pass1)
        first_pass_assignment = {}
        for r, c in zip(row_ind, col_ind):
            first_pass_assignment[logical_qubits[r]] = physical_qubits[c]
    else:
        first_pass_assignment = {}

    # --- Step 4: Second pass - refined cost using first-pass assignment ---
    # cost[i][j] = sum over neighbors k of lq_i: interaction_weight(i,k) * distance_matrix[pq_j][best_physical_for_k]
    cost_matrix_pass2 = np.zeros((n_logical, n_phys))
    for i, lq in enumerate(logical_qubits):
        for j, pq in enumerate(physical_qubits):
            cost = 0.0
            for neighbor_lq, w in logical_neighbors[lq].items():
                if neighbor_lq in first_pass_assignment:
                    best_phys_for_neighbor = first_pass_assignment[neighbor_lq]
                    cost += w * self.distance_matrix[pq][best_phys_for_neighbor]
            cost_matrix_pass2[i][j] = cost

    # --- Step 5: Solve refined assignment ---
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q

    if n_logical > 0:
        row_ind, col_ind = linear_sum_assignment(cost_matrix_pass2)
        for r, c in zip(row_ind, col_ind):
            lq = logical_qubits[r]
            pq = physical_qubits[c]
            mapping_dict[lq] = pq
            reverse_mapping_dict[pq] = lq

    # --- Step 6: Fill unmapped qubits ---
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    free_physical = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]

    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)