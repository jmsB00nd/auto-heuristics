# Idea: Harmonic Inverse Distance Priority Cost (HIDPC)
# Stats: {"mean_swaps": 702.0909090909091, "mean_depth": 1220.5454545454545, "mean_runtime": 1.2004704800519077, "total_circuits": 22, "successful_runs": 22, "failed_runs": 0, "error": null, "first_failure_error": null, "first_failure_traceback": null}

def qlosure_poly_heuristic(self, swap_gate):
    W = 0.5
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front Layer: Weighted Harmonic Mean ---
    # Weighted harmonic mean: H = Σ(w_i) / Σ(w_i / d_i)
    # A single small distance dominates: creates urgency to finish near-executable gates.
    f_weight_sum = 0.0
    f_inv_sum = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        weight = deps + 1
        d = self.distance_matrix[Q1][Q2] + 1  # +1 avoids div-by-zero; d=1 now pulls hardest
        f_weight_sum += weight
        f_inv_sum += weight / d

    # H_front = total_weight / Σ(weight/d); lower is better (dominated by smallest d)
    f_harmonic = f_weight_sum / f_inv_sum if f_inv_sum > 0.0 else 0.0

    # --- Extended Layer: Weighted Harmonic Mean with Layer Decay ---
    # Gates deeper in the lookahead receive lower weights (layer_factor discounts them).
    e_weight_sum = 0.0
    e_inv_sum = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        weight = (deps + 1) / layer_factor  # deeper gates contribute less
        d = self.distance_matrix[Q1][Q2] + 1
        e_weight_sum += weight
        e_inv_sum += weight / d

    e_harmonic = e_weight_sum / e_inv_sum if e_inv_sum > 0.0 else 0.0

    # Combine: decay-weighted sum of front + lookahead harmonic costs
    H = max_decay * (
        f_harmonic
        + W * (e_harmonic if extended_layer_size else 0.0)
    )

    return H