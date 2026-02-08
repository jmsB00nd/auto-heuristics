# Strategy: "Inverse Distance Deficit with Rank-Normalized Criticality (IDD-RNC)"
# Intuition: Adjacent gate pairs (d=1) after a candidate SWAP are immediately executable and should contribute **exactly zero** cost — unlike `d/(d+1)` saturation which still assigns 0.5 to resolved gates, diluting the signal. The `(d-1)/d` deficit function achieves this while still saturating for large distances. Raw dependency counts vary by orders of magnitude across circuits; rank-normalizing criticality into [0,1] creates a scale-invariant urgency weight that adapts to any circuit structure.
# Stats: {'mean_swaps': 752.8636363636364, 'mean_depth': 1212.090909090909, 'mean_runtime': 1.8971310312097722, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # Collect criticality values for rank normalization across the visible window
    all_gates = list(self.front_layer) + list(self.extended_layer)
    if all_gates:
        crit_values = sorted(set(self.dag_dependencies_count[g] for g in all_gates))
        num_ranks = len(crit_values)
        if num_ranks > 1:
            rank_map = {v: i / (num_ranks - 1) for i, v in enumerate(crit_values)}
        else:
            rank_map = {crit_values[0]: 1.0}
    else:
        rank_map = {}

    # Front Layer: Inverse Distance Deficit with Rank Weighting
    # Cost per gate: rank_weight * (d - 1) / d
    # Key: d=1 (adjacent, executable after swap) contributes ZERO cost
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        d = self.distance_matrix[Q1][Q2]

        crit = self.dag_dependencies_count[g]
        rank_w = 0.5 + rank_map.get(crit, 0.5)

        if d > 0:
            f_cost += rank_w * (d - 1.0) / d

    f_norm = f_cost / front_layer_size if front_layer_size else 0.0

    # Extended Layer: Same deficit metric with depth^1.5 decay
    e_cost = 0.0
    if extended_layer_size:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            d = self.distance_matrix[Q1][Q2]
            depth = self.extended_layer_index.get(g, 0) + 1

            crit = self.dag_dependencies_count[g]
            rank_w = 0.5 + rank_map.get(crit, 0.5)

            if d > 0:
                deficit = (d - 1.0) / d
            else:
                deficit = 0.0

            e_cost += rank_w * deficit / (depth ** 1.5)

        e_norm = e_cost / extended_layer_size
    else:
        e_norm = 0.0

    W = 0.5
    H = max_decay * (f_norm + W * e_norm)

    return H