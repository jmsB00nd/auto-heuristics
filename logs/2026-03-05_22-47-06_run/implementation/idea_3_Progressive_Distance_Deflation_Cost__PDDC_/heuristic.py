def qlosure_poly_heuristic(self, swap_gate):
    import math

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Front layer: log-deflated distance weighted by dependency urgency
    # log(1+d): d=1→0.69, d=2→1.10, d=4→1.61, d=8→2.20
    # Nearly-executable gates (d=1,2) contribute minimally vs baseline's linear d
    f_distance = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        d = self.distance_matrix[Q1][Q2]

        log_d = math.log(1 + d)
        f_distance += (deps + 1) * log_d

    # Extended layer: double log-deflation — distance AND layer depth both compressed
    # Discounts future gates sub-linearly in both axes, unlike 1/layer_factor in baseline
    e_distance = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        d = self.distance_matrix[Q1][Q2]

        log_d = math.log(1 + d)
        log_layer = math.log(1 + layer_factor)  # log-compressed depth discount
        e_distance += (deps + 1) * log_d / log_layer

    H = max_decay * (
        f_distance / front_layer_size +
        ((e_distance / extended_layer_size) if extended_layer_size else 0)
    )

    return H