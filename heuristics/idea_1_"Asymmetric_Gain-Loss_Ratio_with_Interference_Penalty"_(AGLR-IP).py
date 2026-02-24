# Strategy: "Asymmetric Gain-Loss Ratio with Interference Penalty" (AGLR-IP)
# Intuition: None provided
# Stats: {'mean_swaps': 683.6818181818181, 'mean_depth': 1122.1363636363637, 'mean_runtime': 1.5869094241749158, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # "Asymmetric Gain-Loss Ratio with Interference Penalty" (AGLR-IP)
    #
    # Core idea: Score a swap by its MARGINAL EFFECT (gain ratio per gate),
    # not its absolute post-swap cost. Penalize "selfish" swaps where
    # improving one front gate worsens another (interference).

    LOSS_PENALTY = 2.3      # Losses hurt more than gains help (asymmetry)
    CRIT_PWR = 0.7          # Sub-linear criticality scaling
    DEPTH_DECAY = 0.6       # How fast extended-layer influence fades
    EXT_WEIGHT = 0.8        # Extended layer contribution
    INTERFERENCE_W = 0.35   # Penalty weight for front-layer interference

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front Layer: Criticality-weighted gain ratios ---
    front_gains = []    # positive weighted gains (improvements)
    front_losses = []   # negative weighted gains (regressions), scaled by LOSS_PENALTY
    f_net = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        # Distance BEFORE this candidate swap (current mapping)
        P1_before = self.mapping_dict[q1]
        P2_before = self.mapping_dict[q2]
        dist_before = self.distance_matrix[P1_before][P2_before]

        # Distance AFTER this candidate swap (hypothetical mapping)
        P1_after = self.temp_mapping_dict[q1]
        P2_after = self.temp_mapping_dict[q2]
        dist_after = self.distance_matrix[P1_after][P2_after]

        # Signed gain ratio: positive = improvement, negative = regression
        # Denominator (dist_before + 1) normalizes so near and far gates
        # contribute proportionally to their fractional improvement
        delta = dist_before - dist_after
        gain_ratio = delta / (dist_before + 1.0)

        crit_weight = (self.dag_dependencies_count[g] + 1.0) ** CRIT_PWR

        weighted_gain = gain_ratio * crit_weight

        if delta >= 0:
            front_gains.append(weighted_gain)
        else:
            # Losses penalized asymmetrically harder (regressions cascade)
            front_losses.append(weighted_gain * LOSS_PENALTY)

        f_net += weighted_gain if delta >= 0 else weighted_gain * LOSS_PENALTY

    # --- Interference penalty ---
    # If both gains AND losses exist in the front layer, this swap is
    # "robbing Peter to pay Paul" — penalize proportional to the conflict
    interference = 0.0
    if front_gains and front_losses:
        total_gain = sum(front_gains)
        total_loss = -sum(front_losses)  # Make positive for comparison
        # Geometric mean of gain and loss magnitudes:
        # high only when both are substantial (true conflict)
        interference = (total_gain * total_loss) ** 0.5

    # --- Extended Layer: Gain ratios with depth decay ---
    e_net = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        P1_before = self.mapping_dict[q1]
        P2_before = self.mapping_dict[q2]
        dist_before = self.distance_matrix[P1_before][P2_before]

        P1_after = self.temp_mapping_dict[q1]
        P2_after = self.temp_mapping_dict[q2]
        dist_after = self.distance_matrix[P1_after][P2_after]

        delta = dist_before - dist_after
        gain_ratio = delta / (dist_before + 1.0)

        deps = self.dag_dependencies_count[g]
        layer = self.extended_layer_index.get(g, 0) + 1.0
        crit_weight = (deps + 1.0) ** CRIT_PWR / (layer ** DEPTH_DECAY)

        if delta >= 0:
            e_net += gain_ratio * crit_weight
        else:
            e_net += gain_ratio * crit_weight * LOSS_PENALTY

    # --- Aggregation ---
    # Negate because higher net gain = better swap, but we minimize H
    front_size = len(self.front_layer) if self.front_layer else 1
    ext_size = len(self.extended_layer) if self.extended_layer else 1

    f_term = f_net / front_size
    e_term = e_net / ext_size

    # More gain → lower H (better); interference adds penalty
    H = -max_decay * (f_term + EXT_WEIGHT * e_term) + INTERFERENCE_W * interference

    return float(H)