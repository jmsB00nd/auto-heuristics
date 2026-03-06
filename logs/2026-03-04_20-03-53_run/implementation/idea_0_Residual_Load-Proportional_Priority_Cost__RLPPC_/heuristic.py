def qlosure_poly_heuristic(self, swap_gate):
    W = 1
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Compute residual gate load per logical qubit across all remaining visible gates.
    # A qubit with high load has many future routing decisions depending on its
    # current placement — it deserves higher priority now.
    qubit_load = {}
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        qubit_load[q1] = qubit_load.get(q1, 0) + 1
        qubit_load[q2] = qubit_load.get(q2, 0) + 1
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        qubit_load[q1] = qubit_load.get(q1, 0) + 1
        qubit_load[q2] = qubit_load.get(q2, 0) + 1

    f_distance = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        # Geometric mean of remaining loads: sqrt(L_q1 * L_q2)
        # Penalizes distance proportionally to how "busy" both qubits remain.
        # Near-retired qubits (load=1) barely inflate cost; high-load pairs dominate.
        geo_mean_load = (qubit_load.get(q1, 1) * qubit_load.get(q2, 1)) ** 0.5
        f_distance += geo_mean_load * self.distance_matrix[Q1][Q2]

    e_distance = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        geo_mean_load = (qubit_load.get(q1, 1) * qubit_load.get(q2, 1)) ** 0.5
        e_distance += geo_mean_load * self.distance_matrix[Q1][Q2] / layer_factor

    H = max_decay * (f_distance / front_layer_size + W *
                    ((e_distance / extended_layer_size) if extended_layer_size else 0))

    return H