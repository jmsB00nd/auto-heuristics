# Strategy: ** Criticality-Adaptive Distance Exponent (CADE)
# Intuition: ** The optimal shape of the distance penalty should depend on gate criticality — high-criticality bottleneck gates deserve a convex (super-linear, ~1.5) distance penalty that harshly punishes any separation, while low-criticality gates tolerate a concave (sub-linear, ~0.5) penalty that avoids wasting swap resources on them. By making the distance exponent itself a smooth function of normalized criticality, the router dynamically allocates routing pressure per-gate rather than using a fixed one-size-fits-all exponent.
# Stats: {'mean_swaps': 657.7727272727273, 'mean_depth': 972.0909090909091, 'mean_runtime': 5.159603660756892, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # --- Hyperparameters ---
    ALPHA = 1.0           # Exponent range: dist^(BASE_EXP) to dist^(BASE_EXP + ALPHA)
    BASE_EXP = 0.5        # Minimum exponent (concave, for low-crit gates)
    GAUSS_2SIG2 = 8.0     # Gaussian depth decay: 2*sigma^2 (sigma=2)
    LOOKAHEAD_W = 0.6     # Extended layer contribution weight

    p1, p2 = swap_gate
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # --- Compute criticality normalization bounds across visible window ---
    # We need min/max to normalize criticality into [0, 1]
    crit_min = 1e18
    crit_max = 0.0

    for g in self.front_layer:
        c = self.dag_dependencies_count[g]
        if c < crit_min:
            crit_min = c
        if c > crit_max:
            crit_max = c

    for g in self.extended_layer:
        c = self.dag_dependencies_count[g]
        if c < crit_min:
            crit_min = c
        if c > crit_max:
            crit_max = c

    crit_range = crit_max - crit_min
    if crit_range <= 0:
        crit_range = 1.0  # Avoid division by zero; all gates equal

    # --- Front Layer: Adaptive-Exponent Distance ---
    f_cost = 0.0
    f_count = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]

        deps = self.dag_dependencies_count[g]
        crit_norm = (deps - crit_min) / crit_range  # In [0, 1]

        # Adaptive exponent: low-crit -> 0.5 (concave), high-crit -> 1.5 (convex)
        exponent = BASE_EXP + ALPHA * crit_norm

        # Criticality weight (sub-linear to prevent runaway)
        crit_w = (deps + 1.0) ** 0.65

        f_cost += crit_w * (dist ** exponent)
        f_count += 1

    # --- Extended Layer: Adaptive Exponent + Gaussian Depth Decay ---
    e_cost = 0.0
    e_count = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]

        deps = self.dag_dependencies_count[g]
        crit_norm = (deps - crit_min) / crit_range

        depth = self.extended_layer_index.get(g, 0)

        # Gaussian depth gate: strong attention on immediate successors
        decay = math.exp(-(depth ** 2) / GAUSS_2SIG2)

        # Adaptive exponent (slightly less aggressive for future gates)
        exponent = BASE_EXP + ALPHA * crit_norm * 0.7

        crit_w = (deps + 1.0) ** 0.65

        e_cost += crit_w * (dist ** exponent) * decay
        e_count += 1

    # --- Normalization & Aggregation ---
    h_f = (f_cost / f_count) if f_count > 0 else 0.0
    h_e = (e_cost / e_count) if e_count > 0 else 0.0

    H = max_decay * (h_f + LOOKAHEAD_W * h_e)
    return float(H)