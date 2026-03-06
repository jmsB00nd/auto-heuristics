def qlosure_poly_heuristic(self, swap_gate):
    W = 1.0
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front layer: weighted harmonic mean of distances ---
    # H_w = sum(w_i) / sum(w_i / d_i)
    # Small distances (d=1,2) dominate the denominator, pulling the mean down
    # and urgently rewarding SWAPs that resolve near-executable gates.
    f_weight_sum = 0.0
    f_inv_weighted_sum = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        d = max(self.distance_matrix[Q1][Q2], 1)  # clamp to 1 to avoid div-by-zero
        w = self.dag_dependencies_count[g] + 1
        f_weight_sum += w
        f_inv_weighted_sum += w / d  # large when d is small → harmonic pulls toward d=1

    # Weighted harmonic mean: a single "representative distance" biased toward closest gates
    f_harmonic = f_weight_sum / f_inv_weighted_sum if f_inv_weighted_sum > 0 else 0.0

    # --- Extended layer: weighted harmonic mean with depth discounting ---
    e_weight_sum = 0.0
    e_inv_weighted_sum = 0.0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        d = max(self.distance_matrix[Q1][Q2], 1)
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        w = (self.dag_dependencies_count[g] + 1) / layer_factor  # deeper gates weighted less
        e_weight_sum += w
        e_inv_weighted_sum += w / d

    e_harmonic = e_weight_sum / e_inv_weighted_sum if e_inv_weighted_sum > 0 else 0.0

    # Normalise by layer sizes (same structural form as baseline) so scales are comparable
    H = max_decay * (
        f_harmonic / front_layer_size
        + W * (e_harmonic / extended_layer_size if extended_layer_size else 0.0)
    )

    return H