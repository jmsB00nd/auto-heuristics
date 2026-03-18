def qlosure_poly_heuristic(self, swap_gate):
    """
    Smooth Exponential Decay with Bottleneck Awareness (SEDBA)

    Key improvements over MSGTC:
    1. Per-gate exponential decay gamma^(depth-1) instead of bucketed decay.
       No precision loss from binning — depth 3 and depth 4 get distinct weights.
    2. Dependency-weighted normalization: sum(w * cost) / sum(w) where
       w = (deps+1) * gamma^(depth-1). Avoids distortion from unequal bucket sizes.
    3. Bottleneck blend for front layer: (1-beta)*mean + beta*max.
       Penalizes any single "stuck" front-layer pair that averaging would mask.
    """
    gamma = 0.55      # per-depth decay; depth-5 weight ≈ 0.092 (smooth roll-off)
    W_ext = 0.42      # extended-layer influence (slightly reduced vs MSGTC since
                      # bottleneck term already captures worst-case front pressure)
    beta  = 0.35      # bottleneck blend: 0 = pure mean, 1 = pure max

    front_size = max(len(self.front_layer), 1)
    max_decay  = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front layer: dependency-weighted mean + bottleneck blend ---
    f_sum      = 0.0
    f_max_cost = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        cost = (deps + 1) * self.distance_matrix[Q1][Q2]
        f_sum += cost
        if cost > f_max_cost:
            f_max_cost = cost

    f_mean  = f_sum / front_size
    f_score = (1.0 - beta) * f_mean + beta * f_max_cost

    # --- Extended layer: per-gate smooth exponential decay,
    #     dependency-weighted normalization to remove size artifacts ---
    e_num = 0.0
    e_den = 0.0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        depth = self.extended_layer_index.get(g, 0) + 1   # 1-based BFS depth
        deps  = self.dag_dependencies_count[g]
        d     = self.distance_matrix[Q1][Q2]

        w      = (deps + 1) * (gamma ** (depth - 1))
        e_num += w * d
        e_den += w

    e_score = e_num / e_den if e_den > 0 else 0.0

    H = max_decay * (f_score + W_ext * e_score)
    return H