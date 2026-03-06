def qlosure_poly_heuristic(self, swap_gate):
    # --- Phase 1: Compute per-logical-qubit interaction statistics ---
    # remaining_gates[q]: total future gates involving logical qubit q
    # first_depth[q]: soonest layer depth at which q next interacts
    remaining_gates = {}
    first_depth = {}

    # Front layer: most imminent, assign depth = 1
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        for q in (q1, q2):
            remaining_gates[q] = remaining_gates.get(q, 0) + 1
            first_depth[q] = min(first_depth.get(q, float('inf')), 1)

    # Extended layer: depth starts at 2 so it never beats front layer
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        depth = self.extended_layer_index.get(g, 0) + 2
        for q in (q1, q2):
            remaining_gates[q] = remaining_gates.get(q, 0) + 1
            first_depth[q] = min(first_depth.get(q, float('inf')), depth)

    # --- Phase 2: Interaction pressure per logical qubit ---
    # pressure(q) = remaining_gates(q) / first_interaction_depth(q)
    # High pressure = many gates AND they arrive soon
    def pressure(q):
        rg = remaining_gates.get(q, 0)
        fd = first_depth.get(q, float('inf'))
        if rg == 0 or fd == float('inf'):
            return 0.0
        return rg / fd

    front_layer_size = max(1, len(self.front_layer))
    extended_layer_size = len(self.extended_layer)

    # --- Phase 3: Pressure-weighted distance cost over front layer ---
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        # Combined pressure: a gate between two high-pressure qubits is most urgent
        combined_pressure = pressure(q1) + pressure(q2)
        f_cost += combined_pressure * self.distance_matrix[Q1][Q2]

    # --- Phase 4: Pressure-weighted distance cost over extended layer ---
    # Decayed by lookahead depth (further gates matter less)
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        combined_pressure = pressure(q1) + pressure(q2)
        e_cost += combined_pressure * self.distance_matrix[Q1][Q2] / layer_factor

    # --- Phase 5: Aggregate with decay penalty ---
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    H = max_decay * (
        f_cost / front_layer_size +
        (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H