# Strategy: **"Minimax Regret with Bottleneck Amplification"**
# Intuition: The router only progresses when at least one front-layer gate reaches adjacency. Instead of averaging distances, explicitly track the minimum distance (nearest-to-executable gate) and the criticality-weighted maximum distance (worst bottleneck). The cost blends a "greedy progress" signal (minimize the min) with a "bottleneck avoidance" signal (minimize the max), weighted by criticality. This prevents the router from myopically chasing one easy gate while creating irrecoverable bottleneck situations.
# Stats: {'mean_swaps': 714.6818181818181, 'mean_depth': 1092.8181818181818, 'mean_runtime': 1.9222660823301836, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    p1, p2 = swap_gate

    # ── Front Layer: Minimax Regret ──
    # Track both the minimum distance (nearest to execution) and
    # the criticality-weighted maximum distance (worst bottleneck).
    front_layer_size = len(self.front_layer)

    min_front_dist = float('inf')
    max_crit_dist = 0.0
    sum_regret = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        P1, P2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[P1][P2]
        deps = self.dag_dependencies_count[g]

        # Criticality weight: square root to dampen extreme values
        crit_w = math.sqrt(deps + 1)

        # Track the closest gate to execution (greedy progress signal)
        min_front_dist = min(min_front_dist, dist)

        # Track the worst criticality-weighted bottleneck
        weighted_dist = crit_w * dist
        max_crit_dist = max(max_crit_dist, weighted_dist)

        # Per-gate regret: saturating function with steep gradient near 0
        # regret(d) = 1 - 1/(1 + d^1.3)  -- zero at d=0, ~1 for large d
        regret = 1.0 - 1.0 / (1.0 + dist ** 1.3)
        sum_regret += crit_w * regret

    if front_layer_size == 0:
        return 0.0

    if min_front_dist == float('inf'):
        min_front_dist = 0

    avg_regret = sum_regret / front_layer_size

    # Blend: average regret + min-distance progress + bottleneck penalty
    # alpha controls greediness (chasing nearest executable gate)
    # beta controls bottleneck aversion
    alpha = 0.4
    beta = 0.3
    front_cost = avg_regret + alpha * min_front_dist + beta * (max_crit_dist / front_layer_size)

    # ── Extended Layer: Exponential-Decay Lookahead ──
    extended_layer_size = len(self.extended_layer)
    e_cost = 0.0

    if extended_layer_size:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            P1, P2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[P1][P2]
            deps = self.dag_dependencies_count[g]
            depth = self.extended_layer_index.get(g, 0) + 1

            crit_w = math.sqrt(deps + 1)

            # Exponential decay: gates further in future matter less
            decay = 0.65 ** depth
            e_cost += crit_w * dist * decay

        e_cost /= extended_layer_size

    # ── Thermal Regularization ──
    # Small additive penalty for hot qubits (tie-breaking + noise avoidance)
    heat = 0.002 * (self.decay_parameter[p1] + self.decay_parameter[p2])

    # ── Final Score: Minimax Regret ──
    W = 0.5
    H = heat + front_cost + W * e_cost

    return H