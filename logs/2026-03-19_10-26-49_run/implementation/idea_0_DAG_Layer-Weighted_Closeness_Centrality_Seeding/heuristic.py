def init_mapping(self):
    from collections import defaultdict

    # --- Step 1: Identify all logical qubits ---
    logical_qubit_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 2: Assign DAG layers to gates ---
    # Process gates in sorted key order as a proxy for topological order.
    # A gate's layer = max(ready_time of all its qubits).
    qubit_ready_time = {}  # logical qubit -> next free layer
    gate_layer = {}
    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        if len(qubits) == 1:
            q = qubits[0]
            layer = qubit_ready_time.get(q, 0)
            gate_layer[gate] = layer
            qubit_ready_time[q] = layer + 1
        elif len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            layer = max(qubit_ready_time.get(q1, 0), qubit_ready_time.get(q2, 0))
            gate_layer[gate] = layer
            qubit_ready_time[q1] = layer + 1
            qubit_ready_time[q2] = layer + 1

    # --- Step 3: Compute temporal centrality for each logical qubit ---
    # Each 2-qubit gate at layer L contributes exp(-ln(2)*L) = 2^(-L) to both
    # its qubits' scores, so layer-0 gates contribute 1.0, layer-1 -> 0.5, etc.
    # This makes the earliest interactions dominate, reflecting that early routing
    # bottlenecks are hardest to resolve with SWAPs.
    temporal_score = defaultdict(float)
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            layer = gate_layer.get(gate, 0)
            weight = math.exp(-math.log(2) * layer)  # 2^{-layer}
            temporal_score[q1] += weight
            temporal_score[q2] += weight

    # Qubits with no 2-qubit interactions stay at score 0 (placed last)
    for q in logical_qubits:
        if q not in temporal_score:
            temporal_score[q] = 0.0

    # --- Step 4: Compute closeness centrality for each physical qubit ---
    # closeness(p) = reachable_count / sum_of_distances(p, all others)
    # Higher value means more central — shorter average path to every other qubit.
    phys_closeness = {}
    for p in physical_qubits:
        total_dist = 0
        reachable = 0
        for o in physical_qubits:
            if o != p:
                d = self.distance_matrix[p][o]
                if d != float('inf'):
                    total_dist += d
                    reachable += 1
        phys_closeness[p] = reachable / total_dist if total_dist > 0 else 0.0

    # --- Step 5: Direct rank-match logical -> physical ---
    # Logical qubit with the highest temporal centrality is mapped to the
    # physical qubit with the highest closeness centrality, and so on.
    sorted_logical = sorted(logical_qubits, key=lambda q: temporal_score[q], reverse=True)
    sorted_physical = sorted(physical_qubits, key=lambda p: phys_closeness[p], reverse=True)

    lq_to_phys = {}
    for i, lq in enumerate(sorted_logical):
        if i < len(sorted_physical):
            lq_to_phys[lq] = sorted_physical[i]

    # --- Step 6: Build strict bijection via swap-chain ---
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

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
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)