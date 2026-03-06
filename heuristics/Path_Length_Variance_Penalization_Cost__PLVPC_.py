def qlosure_poly_heuristic(self, swap_gate):
    W = 0.5      # extended layer blending weight
    ALPHA = 1.0  # variance penalty strength

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front layer ---
    # Collect raw (unweighted) distances for variance, weighted for mean
    raw_f = []
    weighted_f_sum = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        d = self.distance_matrix[Q1][Q2]
        deps = self.dag_dependencies_count[g]
        raw_f.append(d)
        weighted_f_sum += (deps + 1) * d

    f_mean = weighted_f_sum / front_layer_size if front_layer_size else 0.0

    # Variance of raw distances (unweighted) — the core PLVPC signal
    raw_f_mean = sum(raw_f) / front_layer_size if front_layer_size else 0.0
    f_variance = (
        sum((d - raw_f_mean) ** 2 for d in raw_f) / front_layer_size
        if front_layer_size else 0.0
    )

    f_cost = f_mean + ALPHA * f_variance

    # --- Extended layer ---
    raw_e = []
    weighted_e_sum = 0.0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        d_decay = self.distance_matrix[Q1][Q2] / layer_factor
        raw_e.append(d_decay)
        weighted_e_sum += (deps + 1) * d_decay

    if extended_layer_size:
        e_mean = weighted_e_sum / extended_layer_size
        raw_e_mean = sum(raw_e) / extended_layer_size
        e_variance = sum((d - raw_e_mean) ** 2 for d in raw_e) / extended_layer_size
        e_cost = e_mean + ALPHA * e_variance
    else:
        e_cost = 0.0

    H = max_decay * (f_cost + W * e_cost)
    return H