# Strategy: "Polynomial Transitive Bottlenecking (PTB)"
# Intuition: This heuristic treats the circuit as a non-linear pressure field where each gate's "gravitational pull" is a power-law function of its transitive dependency size (the "bottleneck mass"). By scaling the physical distance with a polynomial exponent derived from this mass, the router aggressively prioritizes resolving deep dependency chains while using a quadratic depth-decay to prevent future noise from disrupting immediate mapping requirements.
# Stats: {'mean_swaps': 547.7727272727273, 'mean_depth': 935.5454545454545, 'mean_runtime': 2.177840059453791, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    """
    Calculates a novel cost based on Power-Law Transitive Saliency.
    Lower scores indicate a more optimal SWAP for bottleneck resolution.
    """
    q1, q2 = swap_gate
    
    # 1. Physical Heat Penalty (Tie-breaking and congestion avoidance)
    # We use a small linear factor to bias away from overused physical qubits.
    cost = (self.decay_parameter[q1] + self.decay_parameter[q2]) * 5.0
    
    # 2. Front Layer Contribution (Immediate Constraints)
    # Use a higher polynomial power (1.5) for distance to penalize long-range 
    # separations in gates ready to execute.
    for gate_id in self.front_layer:
        qubits = self.access2q[gate_id]
        if len(qubits) == 2:
            l1, l2 = qubits
            # temp_mapping_dict reflects the state AFTER the potential swap
            p1, p2 = self.temp_mapping_dict[l1], self.temp_mapping_dict[l2]
            dist = self.distance_matrix[p1][p2]
            
            # Saliency is the Transitive Closure size (bottleneck importance)
            # Power-law scaling (0.85) ensures bottlenecks dominate linearly.
            saliency = (self.dag_dependencies_count[gate_id] + 1) ** 0.85
            cost += saliency * (dist ** 1.5)
            
    # 3. Extended Layer Contribution (Lookahead constraints)
    # Use a slightly lower distance power (1.2) and quadratic depth decay.
    for gate_id in self.extended_layer:
        qubits = self.access2q[gate_id]
        if len(qubits) == 2:
            l1, l2 = qubits
            p1, p2 = self.temp_mapping_dict[l1], self.temp_mapping_dict[l2]
            dist = self.distance_matrix[p1][p2]
            
            depth = self.extended_layer_index[gate_id]
            crit = self.dag_dependencies_count[gate_id]
            
            # Quadratic dampening based on lookahead depth
            # Lookahead saliency is dampened compared to front layer (0.6)
            lookahead_weight = ((crit + 1) ** 0.6) / ((depth + 1.2) ** 2)
            cost += lookahead_weight * (dist ** 1.2)
            
    return float(cost)