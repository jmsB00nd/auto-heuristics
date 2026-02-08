# Strategy: Asymmetric Criticality Weighting (ACW)
# Intuition: The router should be highly sensitive to the dependency pressure (criticality) of immediate gates to clear bottlenecks, but should use a dampened (square-root) criticality for the lookahead layer. This prevents the router from being over-reactive to distant future gates while maintaining strong focus on resolving the most critical interactions in the front layer.
# Stats: {'mean_swaps': 574.3636363636364, 'mean_depth': 944.7272727272727, 'mean_runtime': 1.860409433191473, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Retrieve decay penalty for the candidate swap qubits to prevent ping-ponging
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    # Process Front Layer: Use full criticality weighting and quadratic distance
    f_total, f_count = 0.0, 0
    for g in self.front_layer:
        qubits = self.access2q[g]
        if len(qubits) < 2:
            continue
        q1, q2 = qubits
        # Get physical positions after the candidate swap
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        
        # Saliency for front layer is linear with criticality
        saliency = self.dag_dependencies_count[g] + 1.0
        f_total += saliency * (dist ** 2)
        f_count += 1
        
    # Process Extended Layer: Use root-criticality and linear distance for smoother guidance
    e_total, e_count = 0.0, 0
    for g in self.extended_layer:
        qubits = self.access2q[g]
        if len(qubits) < 2:
            continue
        q1, q2 = qubits
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        
        depth = self.extended_layer_index.get(g, 0)
        # Saliency for lookahead is square-rooted to dampen noise from distant gates
        saliency = (self.dag_dependencies_count[g] + 1.0) ** 0.5
        # Harmonic decay for depth (depth+1 to avoid div by zero)
        e_total += (saliency * dist) / (depth + 1.0)
        e_count += 1
        
    # Calculate normalized layer heuristics
    h_f = (f_total / f_count) if f_count > 0 else 0
    h_e = (e_total / e_count) if e_count > 0 else 0
    
    # Combine using a lookahead factor (W=0.5) and the global decay multiplier
    # Lower score = better candidate
    return max_decay * (h_f + 0.5 * h_e)