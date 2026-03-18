def qlosure_poly_heuristic(self, swap_gate):
    W = 1
    M = 10.0
    eps = 1e-9

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    f_score = 0.0
    resolution_bonus = 0.0
    W_total = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        w = deps + 1
        d = self.distance_matrix[Q1][Q2]

        f_score += w * d
        W_total += w
        if d == 1:
            resolution_bonus += w

    f_score /= front_layer_size

    e_score = 0.0
    if extended_layer_size > 0:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            deps = self.dag_dependencies_count[g]
            e_score += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor
        e_score /= extended_layer_size

    H = (max_decay * (f_score + W * e_score)
         - M * resolution_bonus / (W_total + eps))

    return H