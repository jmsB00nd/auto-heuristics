# Strategy: "Polynomial Saliency-Gated Potential (PSGP)"
# Intuition: This function treats the mapping problem as a multi-tiered potential field where interactions exert "tension" on qubits. By using a quadratic distance penalty for the execution front and a sub-quadratic (1.5-power) penalty for the lookahead window, the router aggressively resolves immediate bottlenecks while maintaining a "fluid" guidance for future gates, all weighted by the square-root of each gate's transitive criticality.
# Stats: {'mean_swaps': 553.3181818181819, 'mean_depth': 951.9545454545455, 'mean_runtime': 2.876920526677912, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Retrieve the physical qubits involved in the candidate swap
    p1, p2 = swap_gate
    
    # h_score accumulates the 'tension' of the current mapping configuration
    h_score = 0.0
    
    # 1. Front Layer Tension: High-Priority Quadratic Minimization
    # These gates are ready to execute; we use a d^2 penalty to 
    # force logical partners together as a primary objective.
    for gate_id in self.front_layer:
        logical_qs = self.access2q[gate_id]
        if len(logical_qs) != 2:
            continue
            
        # Get physical locations after the candidate swap from the temp mapping
        phys_q1 = self.temp_mapping_dict[logical_qs[0]]
        phys_q2 = self.temp_mapping_dict[logical_qs[1]]
        
        dist = self.distance_matrix[phys_q1][phys_q2]
        
        # Saliency is the square root of the gate's criticality (transitive closure size).
        # This dampens the influence of massive gates while still prioritizing them.
        saliency = (self.dag_dependencies_count[gate_id] + 1) ** 0.5
        h_score += saliency * (dist ** 2.0)

    # 2. Extended Layer Tension: Predictive 1.5-Power Guidance
    # We use a sub-quadratic power (1.5) and a temporal decay factor
    # to guide qubits toward future interaction zones without over-constraining the front.
    for gate_id in self.extended_layer:
        logical_qs = self.access2q[gate_id]
        if len(logical_qs) != 2:
            continue
            
        phys_q1 = self.temp_mapping_dict[logical_qs[0]]
        phys_q2 = self.temp_mapping_dict[logical_qs[1]]
        
        dist = self.distance_matrix[phys_q1][phys_q2]
        
        # Depth in the lookahead window (0 is the immediate successor)
        depth = self.extended_layer_index[gate_id]
        
        # Temporal decay: Weights drop off linearly with depth
        temporal_weight = 1.0 / (depth + 1.5)
        saliency = ((self.dag_dependencies_count[gate_id] + 1) ** 0.5) * temporal_weight
        
        # Polynomial penalty (1.5) balances between distance minimization and local flexibility
        h_score += saliency * (dist ** 1.5)

    # 3. Hardware 'Heat' Regularization
    # Penalize the use of 'hot' physical qubits (high decay) to prevent local congestion
    # and distribute SWAP errors across the topology.
    # Scaled by a factor of 10.0 to act as a meaningful tie-breaker.
    heat_friction = (self.decay_parameter[p1] + self.decay_parameter[p2]) * 10.0
    
    return h_score + heat_friction