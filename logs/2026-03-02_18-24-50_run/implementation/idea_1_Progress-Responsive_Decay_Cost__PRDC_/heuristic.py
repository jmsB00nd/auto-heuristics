def qlosure_poly_heuristic(self, swap_gate):
    import math

    front_layer_size = max(len(self.front_layer), 1)
    extended_layer_size = len(self.extended_layer)

    # === Step 1: Measure routing progress for this candidate swap ===
    # Compare front-layer total distance before vs. after the swap
    dist_before = 0.0
    dist_after = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        # Before swap (current mapping)
        B1, B2 = self.mapping_dict[q1], self.mapping_dict[q2]
        if B1 >= 0 and B2 >= 0:
            dist_before += self.distance_matrix[B1][B2]
        # After swap (proposed mapping)
        A1, A2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if A1 >= 0 and A2 >= 0:
            dist_after += self.distance_matrix[A1][A2]

    # Normalized progress rate in [-1, 1]:
    #   > 0 → swap reduces distance (good progress)
    #   = 0 → no change (neutral)
    #   < 0 → swap increases distance (regressing)
    progress_rate = (dist_before - dist_after) / (dist_before + 1e-9)
    progress_rate = max(-1.0, min(1.0, progress_rate))

    # === Step 2: Progress-responsive decay exponent ===
    # decay_exp = exp(-λ * progress_rate)
    #   progress_rate =  1.0 → exp(-1.5) ≈ 0.22  (good progress: soft decay, reward)
    #   progress_rate =  0.0 → exp( 0.0) = 1.00  (neutral: same as baseline)
    #   progress_rate = -1.0 → exp(+1.5) ≈ 4.48  (stuck: hard decay, escape penalty)
    lambda_prdc = 1.5
    decay_exp = math.exp(-lambda_prdc * progress_rate)

    p0, p1 = swap_gate[0], swap_gate[1]
    d0 = self.decay_parameter[p0] ** decay_exp
    d1 = self.decay_parameter[p1] ** decay_exp
    modulated_decay = max(d0, d1)

    # === Step 3: Front-layer cost (criticality-weighted distances) ===
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 >= 0 and Q2 >= 0:
            deps = self.dag_dependencies_count[g]
            f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    # === Step 4: Extended lookahead cost (depth-decayed) ===
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 >= 0 and Q2 >= 0:
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            deps = self.dag_dependencies_count[g]
            e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor

    # === Step 5: Final cost ===
    H = modulated_decay * (
        f_distance / front_layer_size
        + (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    return H