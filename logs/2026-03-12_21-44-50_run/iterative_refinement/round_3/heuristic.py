def qlosure_poly_heuristic(self, swap_gate):
    resolve_reward = 2.0   # bonus subtracted per unit weight for dist=1 gates (immediately executable)
    alpha = 0.65           # smooth per-depth exponential decay for extended layer
    W_ext = 0.4            # extended layer weight relative to front layer

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front layer: penalize remaining distance, strongly reward executability ---
    # Normalize by total dependency weight (proper weighted average, not gate count)
    f_distance = 0.0
    f_weight_sum = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        w = deps + 1
        dist = self.distance_matrix[Q1][Q2]

        f_weight_sum += w
        if dist == 1:
            # Gate is immediately executable after this SWAP: strong reward
            f_distance -= resolve_reward * w
        else:
            f_distance += w * dist

    f_score = f_distance / max(f_weight_sum, 1)

    # --- Extended layer: smooth exponential decay, fully weighted average ---
    # Per-gate weight = (deps+1) * alpha^(depth-1), then take weighted mean distance.
    # Avoids the MSGTC bucketing loss of resolution between depths 3 and 4.
    e_wsum = 0.0
    e_norm = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        dist = self.distance_matrix[Q1][Q2]

        w = (deps + 1) * (alpha ** (layer_factor - 1))
        e_wsum += w * dist
        e_norm += w

    e_score = e_wsum / e_norm if e_norm > 0 else 0.0

    H = max_decay * (f_score + W_ext * e_score)
    return H