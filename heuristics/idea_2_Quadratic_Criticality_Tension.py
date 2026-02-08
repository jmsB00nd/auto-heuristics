# Strategy: Quadratic Criticality Tension
# Intuition: Standard linear distance metrics often fail to penalize outliers—single long wires that degrade fidelity—if the average distance is low. By modeling the connection cost as a harmonic potential (distance squared), we impose a super-linear penalty on long-range interactions for critical gates, forcing the router to prioritize compactness for the most dependent qubits and reducing the "elastic energy" of the circuit.
# Stats: {'mean_swaps': 574.0909090909091, 'mean_depth': 940.1363636363636, 'mean_runtime': 2.0385492606596514, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Heuristic Weights
    W_FRONT = 1.0
    W_EXTENDED = 0.5  # Weight for lookahead layer
    
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    # --- Front Layer Cost (Quadratic) ---
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        deps = self.dag_dependencies_count[g]
        
        # Quadratic Penalty: Criticality * Distance^2
        # This penalizes long wires more aggressively than linear distance
        f_cost += (deps + 1) * (dist ** 2)

    front_size = len(self.front_layer)
    f_norm = f_cost / front_size if front_size > 0 else 0

    # --- Extended Layer Cost (Quadratic with Decay) ---
    e_cost = 0.0
    extended_size = len(self.extended_layer)
    
    if extended_size > 0:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            
            dist = self.distance_matrix[Q1][Q2]
            deps = self.dag_dependencies_count[g]
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            
            # Quadratic Penalty discounted by lookahead depth
            e_cost += (deps + 1) * (dist ** 2) * (1.0 / layer_factor)
            
        e_norm = e_cost / extended_size
    else:
        e_norm = 0

    # Total Heuristic Score
    H = max_decay * (W_FRONT * f_norm + W_EXTENDED * e_norm)
    
    return H