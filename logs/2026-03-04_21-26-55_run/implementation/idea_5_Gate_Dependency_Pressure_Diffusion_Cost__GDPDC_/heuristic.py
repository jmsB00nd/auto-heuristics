def qlosure_poly_heuristic(self, swap_gate):
    ALPHA = 0.5  # damping factor for one-level successor diffusion

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    def gate_dist(g):
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        return self.distance_matrix[Q1][Q2]

    def diffused_pressure(g):
        """pressure(g) = d(g) + alpha * sum of successor distances (one-level diffusion)."""
        d_g = gate_dist(g)
        succ_sum = sum(
            gate_dist(s)
            for s in self.dag2q.get(g, set())
            if s in self.access2q
        )
        return d_g + ALPHA * succ_sum

    f_pressure = sum(diffused_pressure(g) for g in self.front_layer)

    e_pressure = 0.0
    for g in self.extended_layer:
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        e_pressure += diffused_pressure(g) / layer_factor

    W = 1.0
    H = max_decay * (
        f_pressure / front_layer_size
        + W * (e_pressure / extended_layer_size if extended_layer_size else 0.0)
    )
    return H