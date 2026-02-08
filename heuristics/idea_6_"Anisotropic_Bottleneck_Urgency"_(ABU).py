# Strategy: "Anisotropic Bottleneck Urgency" (ABU)
# Intuition: This heuristic applies an anisotropic distance penalty: current bottleneck gates are weighted with a super-linear exponent (1.4) to force immediate resolution of dependencies, while future gates use a linear penalty with harmonic depth decay to maintain long-term topological alignment without over-reacting to distant constraints.
# Stats: {'mean_swaps': 494.6818181818182, 'mean_depth': 935.7272727272727, 'mean_runtime': 1.2449741255153308, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    p1, p2 = swap_gate
    score = 0.0
    
    # 1. Heat-Aware Regularization
    # Small penalty for using 'hot' qubits to balance noise and break distance ties
    score += 0.001 * (self.decay_parameter[p1] + self.decay_parameter[p2])

    # 2. Front Layer: Immediate Bottleneck Resolution
    # We apply a super-linear penalty to active gates to ensure the router 
    # prioritizes resolving these constraints over subtle lookahead gains.
    for gate_id in self.front_layer:
        logical_qubits = self.access2q[gate_id]
        if len(logical_qubits) == 2:
            q_a, q_b = logical_qubits
            dist = self.distance_matrix[self.temp_mapping_dict[q_a]][self.temp_mapping_dict[q_b]]
            
            # Saliency is the root of the transitive closure size (criticality)
            # This prevents massive DAGs from creating numerical instability
            saliency = (self.dag_dependencies_count[gate_id] + 1) ** 0.6
            
            # Anisotropic Tension: (dist^1.4) penalizes separated qubits more aggressively
            score += saliency * (dist ** 1.4)

    # 3. Extended Layer: Harmonic Strategic Alignment
    # For future gates, we use a linear distance model with hyperbolic decay
    # relative to their depth in the lookahead window.
    for gate_id in self.extended_layer:
        logical_qubits = self.access2q[gate_id]
        if len(logical_qubits) == 2:
            q_a, q_b = logical_qubits
            dist = self.distance_matrix[self.temp_mapping_dict[q_a]][self.temp_mapping_dict[q_b]]
            
            # Depth 0 is immediate successor, depth 1 is next level, etc.
            depth = self.extended_layer_index[gate_id]
            
            # Harmonic decay ensures distant future gates don't noise-up the immediate decision
            lookahead_factor = 1.0 / (depth + 1.0)
            saliency = (self.dag_dependencies_count[gate_id] + 1) ** 0.6
            
            # Linear distance penalty for lookahead to provide a smooth gradient
            score += saliency * lookahead_factor * dist
            
    return float(score)