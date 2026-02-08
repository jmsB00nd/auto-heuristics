# Strategy: Sigmoidal Gated Bottlenecking (SGB)
# Intuition: SGB addresses "lookahead noise" by dynamically gating the influence of the extended layer based on the immediate progress of the front layer. When front-layer qubits are far from their targets, the heuristic prioritizes immediate bottleneck resolution using quadratic tension; as the front layer converges (indicated by a low average distance), the "gate" opens to allow future interaction needs to influence the routing, preventing short-sighted SWAP decisions that could create future congestion.
# Stats: {'mean_swaps': 546.5, 'mean_depth': 927.5, 'mean_runtime': 2.3294444409283726, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Physical qubits involved in the candidate SWAP
    p1, p2 = swap_gate[0], swap_gate[1]
    # max_decay captures the 'heat' or usage frequency of the physical qubits
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # 1. Front Layer: Weighted Quadratic Tension
    # We use quadratic tension (dist^2) to aggressively penalize large distances for immediate gates.
    f_weighted_sum = 0.0
    f_dist_sum = 0.0
    n_f = len(self.front_layer)
    
    if n_f > 0:
        for g in self.front_layer:
            q1, q2 = self.access2q[g]
            # Mapping state after the candidate swap
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            crit = self.dag_dependencies_count[g] + 1
            
            f_weighted_sum += crit * (dist * dist)
            f_dist_sum += dist
        f_cost = f_weighted_sum / n_f
        avg_f_dist = f_dist_sum / n_f
    else:
        f_cost = 0.0
        avg_f_dist = 0.0

    # 2. Extended Layer: Weighted Linear Tension with Harmonic Depth Decay
    # Future interaction needs are weighted by criticality and decayed by lookahead depth.
    e_cost = 0.0
    n_e = len(self.extended_layer)
    if n_e > 0:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            crit = self.dag_dependencies_count[g] + 1
            # Lookahead depth factor (1-indexed)
            depth = self.extended_layer_index.get(g, 0) + 1
            e_cost += (crit * dist) / depth
        e_cost /= n_e

    # 3. Gating Mechanism: 4 / (4 + avg_f_dist^2)
    # This factor attenuates the influence of the extended layer when the front layer is far from resolved.
    # It ensures the router stays focused on the immediate bottleneck until the path is relatively clear.
    gating = 4.0 / (4.0 + avg_f_dist * avg_f_dist)
    
    # 4. Final Aggregation
    # The cost combines aggressive local tension with a gated lookahead to balance short-term and long-term goals.
    return max_decay * (f_cost + gating * e_cost)