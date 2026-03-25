def init_mapping(self):
    from collections import defaultdict
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # --- Step 1: Collect logical qubits and compute interaction degrees ---
    logical_qubit_set = set()
    interaction_degree = defaultdict(float)

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            interaction_degree[q1] += 1
            interaction_degree[q2] += 1

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback to trivial identity mapping if circuit has no gates
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 2: Compute normalized degrees ---
    # Logical: interaction degree (2-qubit gate count per qubit)
    log_degrees = np.array(
        [interaction_degree[lq] for lq in logical_qubits], dtype=float
    )
    max_log_deg = log_degrees.max() if log_degrees.max() > 0 else 1.0
    norm_log = log_degrees / max_log_deg  # shape: (n_logical,)

    # Physical: hardware connectivity degree
    hw_degrees = np.array(
        [len(self.backend[pq]) for pq in physical_qubits], dtype=float
    )
    max_hw_deg = hw_degrees.max() if hw_degrees.max() > 0 else 1.0
    norm_hw = hw_degrees / max_hw_deg  # shape: (n_physical,)

    # --- Step 3: Build cost matrix C[l][p] ---
    # C[l][p] = |norm_log[l] - norm_hw[p]|
    # High-degree logical qubits should map to high-degree physical qubits.
    # Minimising this mismatch globally is the assignment objective.
    # Shape: (n_logical, n_physical)
    cost_matrix = np.abs(norm_log[:, np.newaxis] - norm_hw[np.newaxis, :])

    # --- Step 4: Solve linear assignment (Hungarian algorithm, O(n³)) ---
    # scipy handles rectangular matrices: assigns each row to a distinct column
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # --- Step 5: Build logical -> physical map from the optimal assignment ---
    lq_to_phys = {
        logical_qubits[r]: physical_qubits[c]
        for r, c in zip(row_ind, col_ind)
    }

    # --- Step 6: Materialise as a strict bijective permutation of length num_qubits ---
    # Start from identity; apply each target assignment via in-place swaps so that
    # the permutation stays valid (1-to-1) throughout.
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        # The logical qubit currently sitting at target_phys must be displaced
        displaced_lq = reverse_mapping_dict[target_phys]
        # Swap in both directions to preserve bijectivity
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)