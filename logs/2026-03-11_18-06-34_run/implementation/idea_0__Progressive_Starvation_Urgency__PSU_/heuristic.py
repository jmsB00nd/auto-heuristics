def qlosure_poly_heuristic(self, swap_gate):
    W = 0.5
    ALPHA = 0.1  # urgency growth rate per starvation step

    extended_layer_size = len(self.extended_layer)
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front layer: starvation-weighted urgency score ---
    W_total = 0.0
    f_distance = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]

        # Base weight + exponential starvation urgency (capped to avoid overflow)
        base_w = deps + 1
        starvation_steps = self.starvation.get(g, 0)
        urgency = base_w * math.exp(min(ALPHA * starvation_steps, 700.0))

        W_total += urgency
        f_distance += urgency * self.distance_matrix[Q1][Q2]

    # W_total > 0 guaranteed: front_layer non-empty, urgency always > 0
    f_score = f_distance / W_total

    # --- Extended layer: baseline depth-discounted score ---
    e_score = 0.0
    if extended_layer_size > 0:
        e_distance = 0.0
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            deps = self.dag_dependencies_count[g]
            e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor

        e_score = e_distance / extended_layer_size

    H = max_decay * (f_score + W * e_score)
    return H