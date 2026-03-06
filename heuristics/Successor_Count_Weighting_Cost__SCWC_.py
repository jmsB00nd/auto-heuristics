def qlosure_poly_heuristic(self, swap_gate):
    W = 1
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    f_distance = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        # Forward-looking weight: gates that unlock many successors are high-priority
        successors = len(self.dag2q.get(g, set()))
        weight = successors + 1  # +1 keeps terminal gates (0 successors) contributing
        f_distance += weight * self.distance_matrix[Q1][Q2]

    e_distance = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        successors = len(self.dag2q.get(g, set()))
        weight = successors + 1
        e_distance += weight * self.distance_matrix[Q1][Q2] * (1 / layer_factor)

    H = max_decay * (f_distance / front_layer_size + W *
                    ((e_distance / extended_layer_size) if extended_layer_size else 0))

    return H