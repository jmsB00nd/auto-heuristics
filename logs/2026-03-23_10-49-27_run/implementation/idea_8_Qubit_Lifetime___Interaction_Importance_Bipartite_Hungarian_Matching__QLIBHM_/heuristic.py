def init_mapping(self):
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    # --- 1. Collect all logical qubits ---
    logical_qubits_set = set()
    for gate_qubits in self.access.values():
        for q in gate_qubits:
            logical_qubits_set.add(q)
    logical_qubits = sorted(logical_qubits_set)
    num_logical = len(logical_qubits)
    num_physical = self.num_qubits

    # --- 2. Assign a layer to each gate via forward pass ---
    # Gates are sorted by key to approximate topological/program order.
    # Layer of a gate = max(last layer of each of its qubits) + 1.
    qubit_latest_layer = {q: -1 for q in logical_qubits}
    gate_layer = {}
    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        layer = (max(qubit_latest_layer.get(q, -1) for q in qubits) + 1) if qubits else 0
        gate_layer[gate] = layer
        for q in qubits:
            if layer > qubit_latest_layer.get(q, -1):
                qubit_latest_layer[q] = layer

    # --- 3. Compute first/last layer per logical qubit ---
    qubit_first = {}
    qubit_last  = {}
    for gate, layer in gate_layer.items():
        for q in self.access[gate]:
            if q not in qubit_first:
                qubit_first[q] = layer
                qubit_last[q]  = layer
            else:
                if layer < qubit_first[q]: qubit_first[q] = layer
                if layer > qubit_last[q]:  qubit_last[q]  = layer

    # --- 4. Interaction degree (2-qubit gates only) ---
    interaction_degree = {q: 0 for q in logical_qubits}
    for gate_qubits in self.access.values():
        if len(gate_qubits) == 2:
            interaction_degree[gate_qubits[0]] += 1
            interaction_degree[gate_qubits[1]] += 1

    # --- 5. Sustained importance = lifetime × interaction_degree ---
    sustained_importance = {}
    for q in logical_qubits:
        lifetime = qubit_last.get(q, 0) - qubit_first.get(q, 0) + 1
        sustained_importance[q] = lifetime * interaction_degree[q]

    # --- 6. Hardware flexibility = degree in backend graph ---
    physical_degree = {p: len(self.backend.get(p, [])) for p in range(num_physical)}

    # --- 7. Normalize both scores to [0, 1] ---
    si_max  = max(sustained_importance.values()) or 1
    deg_max = max(physical_degree.values())      or 1
    norm_si  = {q: sustained_importance[q] / si_max  for q in logical_qubits}
    norm_deg = {p: physical_degree[p]      / deg_max for p in range(num_physical)}

    # --- 8. Build cost matrix (num_logical × num_physical) ---
    # Minimize mismatch: high-importance logicals → high-flexibility physicals.
    cost_matrix = np.zeros((num_logical, num_physical))
    for i, lq in enumerate(logical_qubits):
        for j in range(num_physical):
            cost_matrix[i, j] = abs(norm_si[lq] - norm_deg[j])

    # --- 9. Hungarian algorithm (optimal assignment) ---
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # --- 10. Populate mapping lists ---
    max_logical_idx = max(logical_qubits) + 1 if logical_qubits else num_physical
    self.mapping_dict         = [-1] * max(max_logical_idx, num_physical)
    self.reverse_mapping_dict = [-1] * num_physical

    for i, j in zip(row_ind, col_ind):
        lq = logical_qubits[i]
        pq = j
        self.mapping_dict[lq]  = pq
        self.reverse_mapping_dict[pq] = lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)