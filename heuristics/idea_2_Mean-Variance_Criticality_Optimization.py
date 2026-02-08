# Strategy: Mean-Variance Criticality Optimization
# Intuition: Circuit execution time is often gated by the "stragglers" (outliers with high distance). Simply minimizing the average distance can mask these bottlenecks. By penalizing the *standard deviation* of the weighted distances alongside the mean, this heuristic drives the system towards a state where *all* critical dependencies are uniformly close, eliminating the worst-case placements that inflate circuit depth.
# Stats: {'mean_swaps': 627.0, 'mean_depth': 949.8636363636364, 'mean_runtime': 2.3995495601133867, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Strategy: Mean-Variance Criticality Optimization
    # Goal: Minimize the statistical upper bound (Mean + StdDev) of the front layer distances
    # to compress the distribution and eliminate high-distance outliers.

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Front Layer: Collect weighted distances
    f_values = []
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        deps = self.dag_dependencies_count[g] + 1
        
        # Weighted distance for this gate
        f_values.append(dist * deps)

    f_score = 0.0
    if f_values:
        n = len(f_values)
        mean = sum(f_values) / n
        if n > 1:
            # Calculate standard deviation to measure spread/outliers
            variance = sum((x - mean) ** 2 for x in f_values) / n
            std_dev = variance ** 0.5
            # Penalize the "upper tail" of the distribution
            f_score = mean + std_dev
        else:
            f_score = mean

    # Extended Layer: Standard Weighted Mean (Lookahead)
    # We use a simple mean here as lookahead variance is less reliable/stable
    e_score = 0.0
    extended_layer_size = len(self.extended_layer)
    if extended_layer_size > 0:
        sum_e = 0.0
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            
            dist = self.distance_matrix[Q1][Q2]
            deps = self.dag_dependencies_count[g] + 1
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            
            sum_e += (dist * deps) / layer_factor
        e_score = sum_e / extended_layer_size

    W = 0.5 # Weight for lookahead
    
    # Total score scaled by the heat of the swap qubits themselves
    H = max_decay * (f_score + W * e_score)

    return H