# Strategy: Relative Saliency Index with Cubic Tension (RSI-CT)
# Intuition: This heuristic introduces a "Relative Saliency Index" that normalizes gate criticality (transitive closure size) against the maximum found in the current lookahead window, preventing deep paths from completely drowning out parallel circuit width. It employs a cubic distance penalty for the front layer to aggressively resolve immediate bottlenecks, while using a 1.5-power depth decay for the extended layer to maintain a balanced pull from future interactions.
# Stats: {'mean_swaps': 520.6818181818181, 'mean_depth': 905.3181818181819, 'mean_runtime': 2.589539549567483, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    f_size = len(self.front_layer)
    e_size = len(self.extended_layer)
    
    if f_size == 0:
        return 0
    
    # 1. Compute Window-Relative Criticality Normalization
    # By normalizing criticality, we make the heuristic more robust across
    # different circuit depths and structures (narrow vs. wide).
    max_crit = 0
    for g in self.front_layer:
        max_crit = max(max_crit, self.dag_dependencies_count[g])
    for g in self.extended_layer:
        max_crit = max(max_crit, self.dag_dependencies_count[g])
    
    # Safe denominator to avoid division by zero
    norm_factor = max_crit + 1.0
    
    # 2. Front Layer: Cubic Tension
    # A cubic penalty (dist^3) creates an extremely steep cost gradient for 
    # distant logical qubits, forcing the router to prioritize these gates 
    # with much higher urgency than a quadratic model.
    f_cost = 0
    for g in self.front_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        q1, q2 = qs
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        # Relative Saliency Index (RSI)
        saliency = (self.dag_dependencies_count[g] + 1.0) / norm_factor
        f_cost += saliency * (self.distance_matrix[Q1][Q2] ** 3)

    # 3. Extended Layer: Fractional Power Depth Decay
    # We use a 1.5-power decay for depth, which sits between harmonic (1.0) 
    # and quadratic (2.0) decay, providing a more stable guidance signal
    # from the lookahead window without causing sudden jumps in cost.
    e_cost = 0
    for g in self.extended_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        q1, q2 = qs
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        depth = self.extended_layer_index.get(g, 0) + 1.0
        saliency = (self.dag_dependencies_count[g] + 1.0) / norm_factor
        
        # Weighted lookahead contribution
        e_cost += (saliency * self.distance_matrix[Q1][Q2]) / (depth ** 1.5)

    # 4. Aggregation and Physical Qubit Heat Penalty
    # We multiply by the max decay (heat) of the physical qubits in the swap 
    # to penalize moves that concentrate logical operations on congested 
    # areas of the QPU topology.
    h_front = f_cost / f_size
    h_ext = (e_cost / e_size) if e_size > 0 else 0
    
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    return max_decay * (h_front + h_ext)