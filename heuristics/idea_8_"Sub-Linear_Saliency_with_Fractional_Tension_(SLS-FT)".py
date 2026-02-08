# Strategy: "Sub-Linear Saliency with Fractional Tension (SLS-FT)"
# Intuition: This function balances dependency criticality by using a sub-linear power (0.77) to prevent massive transitive closures from overwhelming the decision, while employing a fractional-order tension (1.37) on distances to create a smooth but non-linear gradient that favors logical qubit proximity without the extreme sensitivity of quadratic functions.
# Stats: {'mean_swaps': 718.0, 'mean_depth': 991.4545454545455, 'mean_runtime': 2.816455429250544, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Base weights and layer sizes
    W = 1.0
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # Hardware-aware decay/heat parameter for the physical qubits being swapped
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # 1. Front Layer Contribution: Immediate routing pressure
    f_tension = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        
        # Sub-linear saliency (0.77) to dampen the influence of gates with very high dependency counts
        # Fractional-order tension (1.37) to provide a unique non-linear locality gradient
        deps = self.dag_dependencies_count[g]
        saliency = (deps + 1) ** 0.77
        f_tension += saliency * (dist ** 1.37)

    f_score = f_tension / front_layer_size if front_layer_size > 0 else 0

    # 2. Extended Layer Contribution: Future-guided lookahead
    e_tension = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        
        # Distance to the gate in the lookahead window
        depth = self.extended_layer_index.get(g, 0)
        deps = self.dag_dependencies_count[g]
        
        # Hyperbolic depth decay (square-root based) to maintain lookahead relevance
        # while ensuring immediate needs remain dominant.
        depth_decay = 1.0 / ((depth + 1) ** 0.5)
        saliency = (deps + 1) ** 0.77
        e_tension += saliency * (dist ** 1.37) * depth_decay

    e_score = e_tension / extended_layer_size if extended_layer_size > 0 else 0

    # 3. Final Heuristic Combination
    # Lower H = higher priority swap
    H = max_decay * (f_score + W * e_score)

    return float(H)