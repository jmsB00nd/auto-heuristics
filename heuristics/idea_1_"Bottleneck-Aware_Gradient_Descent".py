# Strategy: "Bottleneck-Aware Gradient Descent"
# Intuition: Instead of summing distances uniformly, identify the *bottleneck* gate (highest distance × criticality product) in the front layer and optimize primarily for it, while using the extended layer as a tiebreaker weighted by inverse squared depth. The insight is that the routing problem is often dominated by a single hard-to-resolve interaction — reducing the worst case by 1 swap is more valuable than reducing two easy cases by 1 each, because the hard gate blocks more of the DAG.
# Stats: {'mean_swaps': 545.6818181818181, 'mean_depth': 912.5, 'mean_runtime': 2.07332444190979, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Front layer: bottleneck-focused scoring
    # Track both the max weighted distance and a sum for tiebreaking
    bottleneck = 0.0
    f_sum = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        crit = self.dag_dependencies_count[g] + 1
        weighted = crit * dist
        if weighted > bottleneck:
            bottleneck = weighted
        f_sum += dist

    front_layer_size = len(self.front_layer)
    # Blend: heavily favor the bottleneck, with a small contribution
    # from the average to break ties among equal bottlenecks
    alpha = 0.15
    f_score = bottleneck + alpha * (f_sum / front_layer_size)

    # Extended layer: inverse-square depth decay
    e_score = 0.0
    extended_layer_size = len(self.extended_layer)
    if extended_layer_size:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            depth = self.extended_layer_index.get(g, 0) + 1
            crit = self.dag_dependencies_count[g] + 1
            e_score += crit * self.distance_matrix[Q1][Q2] / (depth * depth)
        e_score /= extended_layer_size

    W = 0.5
    H = max_decay * (f_score + W * e_score)

    return H