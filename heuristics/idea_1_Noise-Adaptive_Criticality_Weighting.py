# Strategy: Noise-Adaptive Criticality Weighting
# Intuition: Standard heuristics treat all qubits as equal nodes in a graph, but real hardware has varying noise levels. This cost function introduces a "Quality-Distance Product" that penalizes placing critical gates on "hot" (high-decay) physical qubits. By multiplying the distance cost by the specific noise levels of the target qubits, it forces high-dependency operations onto the most reliable hardware paths, sacrificing pure hop-count efficiency for higher circuit fidelity.
# Stats: {'mean_swaps': 516.5454545454545, 'mean_depth': 914.1363636363636, 'mean_runtime': 2.425933664495295, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Strategy: Noise-Adaptive Criticality Weighting
    # Goals: 
    # 1. Minimize distance (standard).
    # 2. Maximize hardware reliability for critical gates (novel).
    
    # "Gatekeeper" penalty: Is the swap gate itself on a hot qubit?
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        deps = self.dag_dependencies_count[g] + 1
        
        # Novelty: Noise Factor
        # Calculate the "health" of the physical location where the gate is mapped.
        # If Q1 or Q2 have high decay values, this interaction is 'expensive' in terms of fidelity.
        # We amplify the distance penalty by this noise factor.
        interaction_noise = self.decay_parameter[Q1] + self.decay_parameter[Q2]
        
        # Factor = 1.0 (base) + Noise Penalty
        # We multiply by deps to ensure CRITICAL gates get the cleanest qubits.
        noise_factor = 1.0 + (interaction_noise * 3.0) 
        
        f_cost += (deps * dist * noise_factor)

    e_cost = 0.0
    extended_layer_size = len(self.extended_layer)
    if extended_layer_size > 0:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            
            dist = self.distance_matrix[Q1][Q2]
            deps = self.dag_dependencies_count[g] + 1
            layer_depth = self.extended_layer_index.get(g, 0) + 1
            
            # Gentler noise awareness for lookahead (uncertainty is higher)
            interaction_noise = self.decay_parameter[Q1] + self.decay_parameter[Q2]
            noise_factor = 1.0 + interaction_noise
            
            e_cost += (deps * dist * noise_factor) / layer_depth

    # Normalization
    f_val = f_cost / len(self.front_layer)
    e_val = e_cost / extended_layer_size if extended_layer_size else 0

    W = 0.5
    
    # Total H score: Distance/Quality Metric scaled by the Swap's own risk
    H = max_decay * (f_val + W * e_val)

    return H