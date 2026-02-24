# Strategy: ** Shortest-Path Alignment Pressure (SPAP)
# Intuition: ** Instead of only measuring whether a swap reduces distances, SPAP measures whether a swap lies *on* the shortest path between gate operands. A swap that sits on multiple shortest paths simultaneously is a "bottleneck relief" — it advances multiple gates toward resolution in a topologically efficient direction. This differs from gain-loss heuristics because two swaps may produce identical distance reductions, yet only one follows the natural flow of the coupling graph's shortest-path structure. By combining on-path alignment with criticality-weighted Gaussian lookahead, SPAP preferentially selects swaps that serve as shared routing corridors.

**
# Stats: {'mean_swaps': 776.3181818181819, 'mean_depth': 1081.8181818181818, 'mean_runtime': 4.670216256921941, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # Parameters
    CRIT_PWR = 0.75        # Sub-linear criticality scaling
    GAUSS_SIGMA_SQ2 = 10.0 # 2*sigma^2 for Gaussian depth decay
    ALIGN_BONUS = 0.6      # Weight for path-alignment reward
    LOOKAHEAD_W = 0.8      # Extended layer weight
    HEAT_SCALE = 0.5       # How much local heat matters per-gate

    p1, p2 = swap_gate
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # --- Front Layer ---
    f_cost = 0.0
    f_align = 0.0
    f_count = 0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist_after = self.distance_matrix[Q1][Q2]
        crit = (self.dag_dependencies_count[g] + 1.0) ** CRIT_PWR

        # Base cost: criticality-weighted distance
        local_heat = (self.decay_parameter[Q1] + self.decay_parameter[Q2]) / 2.0
        f_cost += crit * dist_after * (1.0 + HEAT_SCALE * (local_heat - 1.0))

        # Path alignment: check if swap edge lies on the shortest path
        # between Q1 and Q2. If dist(Q1,p1)+1+dist(p2,Q2) == dist(Q1,Q2)
        # or the symmetric version, then swap is on a shortest path.
        if dist_after > 0:
            via_p1p2 = min(
                self.distance_matrix[Q1][p1] + 1 + self.distance_matrix[p2][Q2],
                self.distance_matrix[Q1][p2] + 1 + self.distance_matrix[p1][Q2]
            )
            # Alignment = how close swap is to being on a shortest path (0 = perfect)
            alignment_gap = via_p1p2 - dist_after
            # Negative reward: on-path swaps get bonus (reduce cost)
            f_align += crit * (1.0 / (1.0 + alignment_gap))

        f_count += 1

    # --- Extended Layer ---
    e_cost = 0.0
    e_align = 0.0
    e_count = 0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist_after = self.distance_matrix[Q1][Q2]
        crit = (self.dag_dependencies_count[g] + 1.0) ** CRIT_PWR
        depth = self.extended_layer_index.get(g, 0)

        # Gaussian depth decay
        decay = math.exp(-(depth ** 2) / GAUSS_SIGMA_SQ2)

        e_cost += crit * dist_after * decay

        if dist_after > 0:
            via_p1p2 = min(
                self.distance_matrix[Q1][p1] + 1 + self.distance_matrix[p2][Q2],
                self.distance_matrix[Q1][p2] + 1 + self.distance_matrix[p1][Q2]
            )
            alignment_gap = via_p1p2 - dist_after
            e_align += crit * decay * (1.0 / (1.0 + alignment_gap))

        e_count += 1

    # Normalize
    h_f = (f_cost / f_count) if f_count > 0 else 0.0
    h_e = (e_cost / e_count) if e_count > 0 else 0.0
    a_f = (f_align / f_count) if f_count > 0 else 0.0
    a_e = (e_align / e_count) if e_count > 0 else 0.0

    # Combine: distance cost (minimize) - alignment bonus (maximize → subtract)
    cost_term = h_f + LOOKAHEAD_W * h_e
    align_term = a_f + LOOKAHEAD_W * a_e

    H = max_decay * (cost_term - ALIGN_BONUS * align_term)
    return float(H)