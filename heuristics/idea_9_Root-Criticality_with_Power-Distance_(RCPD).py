# Strategy: Root-Criticality with Power-Distance (RCPD)
# Intuition: This heuristic balances the influence of deep dependency chains by using a square-root scaling for criticality, while employing a 1.5-power distance penalty to aggressively prioritize resolving long-range bottlenecks in the front layer. It uses a quadratic depth decay for the lookahead to focus the router on imminent interactions, creating a sharper decision horizon than harmonic decay.
# Stats: {'mean_swaps': 547.9090909090909, 'mean_depth': 927.8181818181819, 'mean_runtime': 1.8233835805546155, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Calculate the decay factor for the physical qubits involved in the SWAP
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    # Process the front layer (immediate gates)
    f_score = 0.0
    for g in self.front_layer:
        # access2q returns logical qubits; temp_mapping_dict contains physical locations after SWAP
        q_pair = self.access2q[g]
        if not q_pair: continue  # Skip single-qubit gates
        
        q1, q2 = q_pair
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        crit = self.dag_dependencies_count[g]
        
        # Root-criticality prevents deep paths from overwhelming the distance metric
        # Power-law distance (1.5) makes the router very sensitive to long-range separation
        f_score += (crit + 1)**0.5 * (dist ** 1.5)
        
    # Process the extended layer (lookahead)
    e_score = 0.0
    for g in self.extended_layer:
        q_pair = self.access2q[g]
        if not q_pair: continue
        
        q1, q2 = q_pair
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        depth = self.extended_layer_index.get(g, 0)
        crit = self.dag_dependencies_count[g]
        
        # Quadratic depth decay (1/(d+1)^2) creates a sharper horizon than linear or harmonic decay
        e_score += (crit + 1)**0.5 * dist / ((depth + 1) ** 2)
    
    f_size = len(self.front_layer)
    e_size = len(self.extended_layer)
    
    # Normalize by layer sizes and combine with the decay parameter
    # A higher score is worse (we want to minimize distance and criticality-weighted tension)
    f_contribution = f_score / f_size if f_size > 0 else 0
    e_contribution = e_score / e_size if e_size > 0 else 0
    
    return max_decay * (f_contribution + e_contribution)