def qlosure_poly_heuristic(self, swap_gate):
    gamma = 0.55    # continuous per-depth decay; slightly less aggressive than MSGTC's 0.5
    W = 0.4         # extended-layer weight

    front_layer_size = max(len(self.front_layer), 1)
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Front layer: dependency-weighted remaining swap count
    # dist=1 -> adjacent, 0 swaps needed; dist=d -> d-1 swaps needed
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        remaining = max(0.0, self.distance_matrix[Q1][Q2] - 1)
        f_cost += (deps + 1) * remaining
    f_cost /= front_layer_size

    # Extended layer: continuous geometric decay per gate (no coarse bucketing)
    # Normalize by total geometric weight for a properly-scaled weighted average
    e_cost = 0.0
    e_weight_total = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        remaining = max(0.0, self.distance_matrix[Q1][Q2] - 1)
        w = gamma ** (layer_factor - 1)
        e_cost += w * (deps + 1) * remaining
        e_weight_total += w

    if e_weight_total > 0:
        e_cost /= e_weight_total

    H = max_decay * (f_cost + W * e_cost)
    return H