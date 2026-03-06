def qlosure_poly_heuristic(self, swap_gate):
    W = 1
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # --- Coupling Graph Degree-Ratio Peripheral Penalty ---
    # Compute degree of each physical qubit from the backend adjacency list.
    # Peripheral qubits (low degree, e.g. chain endpoints in heavy-hex)
    # are harder to route through and act as routing dead-ends.
    degrees = {q: len(neighbors) for q, neighbors in self.backend.items()}
    max_degree = max(degrees.values()) if degrees else 1

    def peripheral_weight(Q):
        """
        Degree-ratio peripheral penalty weight.
        Central qubits (deg == max_degree)  -> weight = 1.0  (no penalty)
        Peripheral qubits (deg << max_degree) -> weight >> 1  (strong penalty)
        """
        deg = degrees.get(Q, 1)
        return max_degree / deg  # ratio in [1, max_degree]

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front layer: penalise distance by peripheral weight of gate qubits ---
    f_cost = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]

        # Geometric mean of the two peripheral weights captures the combined
        # "dead-end risk" of both qubits without one dominating.
        periph = (peripheral_weight(Q1) * peripheral_weight(Q2)) ** 0.5

        f_cost += (deps + 1) * self.distance_matrix[Q1][Q2] * periph

    # --- Extended layer: same penalty, discounted by lookahead depth ---
    e_cost = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]

        periph = (peripheral_weight(Q1) * peripheral_weight(Q2)) ** 0.5

        e_cost += (deps + 1) * self.distance_matrix[Q1][Q2] * periph / layer_factor

    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0)
    )

    return H