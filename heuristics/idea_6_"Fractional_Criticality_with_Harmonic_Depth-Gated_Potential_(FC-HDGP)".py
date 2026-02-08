# Strategy: "Fractional Criticality with Harmonic Depth-Gated Potential (FC-HDGP)"
# Intuition: This function treats the mapping problem as a dynamic potential field where gate "saliency" is governed by a fractional power-law of the dependency graph (dampening the impact of extreme bottlenecks), while physical distance is penalized with a high-order exponent (1.88) to aggressively minimize multi-hop routing. Lookahead influence is decayed using a harmonic power of its depth (1.62) to maintain focus on the immediate front layer without ignoring future topological constraints.
# Stats: {'mean_swaps': 766.2272727272727, 'mean_depth': 1025.6363636363637, 'mean_runtime': 2.7461383776231245, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    # Physical qubits involved in the candidate swap
    q1_phys, q2_phys = swap_gate
    
    # Qubit "Health" term: Logarithmic average to penalize high-decay physical qubits non-linearly.
    # This prevents the router from overusing noisy or high-traffic qubits.
    thermal_weight = (math.log(self.decay_parameter[q1_phys] + 1.0) + 
                      math.log(self.decay_parameter[q2_phys] + 1.0)) / 2.0
    
    # Heuristic parameters for non-linear weighting
    ALPHA = 0.72  # Scaling factor for gate criticality (transitive closure size)
    BETA = 1.88   # Exponent for physical distance to penalize long-range interactions
    GAMMA = 1.62  # Decay exponent for gates in the lookahead window (extended layer)
    W_LOOKAHEAD = 0.58  # Weighting constant for future operations
    
    f_total = 0.0
    for g_id in self.front_layer:
        # Get logical qubits and their new physical locations after the swap
        l_q1, l_q2 = self.access2q[g_id]
        p1, p2 = self.temp_mapping_dict[l_q1], self.temp_mapping_dict[l_q2]
        dist = self.distance_matrix[p1][p2]
        
        # Saliency represents the "pressure" to resolve a specific gate based on its dependencies
        saliency = (self.dag_dependencies_count[g_id] + 1) ** ALPHA
        # Strong distance penalty for gates ready for execution
        f_total += saliency * (dist ** BETA)
        
    e_total = 0.0
    for g_id in self.extended_layer:
        l_q1, l_q2 = self.access2q[g_id]
        p1, p2 = self.temp_mapping_dict[l_q1], self.temp_mapping_dict[l_q2]
        dist = self.distance_matrix[p1][p2]
        
        # Depth is the index in the lookahead window (0 = immediate successor)
        depth = self.extended_layer_index.get(g_id, 0) + 1
        saliency = (self.dag_dependencies_count[g_id] + 1) ** ALPHA
        
        # Future gate influence decays according to the harmonic power-law of its depth
        e_total += (saliency * dist) / (depth ** GAMMA)
        
    # Calculate averages to normalize against varying layer sizes
    n_front = len(self.front_layer)
    n_ext = len(self.extended_layer)
    
    h_front = f_total / n_front if n_front > 0 else 0.0
    h_ext = e_total / n_ext if n_ext > 0 else 0.0
    
    # Final cost score H. Lower scores indicate more efficient SWAP choices.
    H = thermal_weight * (h_front + W_LOOKAHEAD * h_ext)
    
    return float(H)