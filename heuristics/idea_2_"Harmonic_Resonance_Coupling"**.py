# Strategy: "Harmonic Resonance Coupling"**
# Intuition: ** Physical qubits involved in a SWAP should ideally create a "resonance" effect where moving one qubit closer to its target simultaneously improves the positioning of multiple gate pairs. We measure this by computing a harmonic mean of distance improvements weighted by gate criticality, which naturally penalizes SWAPs that help one gate but harm others (unlike arithmetic means which can hide such trade-offs).
# Stats: {'mean_swaps': 721.7555555555556, 'mean_depth': 1270.4444444444443, 'mean_runtime': 1.0067678531010946, 'total_circuits': 90, 'successful_runs': 90, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    """
    Harmonic Resonance Coupling Heuristic
    
    Uses harmonic mean of inverse distances to naturally penalize SWAPs
    that create distance asymmetries. The harmonic mean is sensitive to
    small values, so a SWAP that brings ANY gate pair far apart will
    dominate the cost, preventing myopic optimizations.
    """
    p0, p1 = swap_gate
    
    # Decay factor for the physical qubits involved
    decay_cost = self.decay_parameter[p0] + self.decay_parameter[p1]
    
    # Compute harmonic coupling score for front layer (highest priority)
    front_layer_harmonic = 0.0
    front_layer_count = 0
    
    for gate_id in self.front_layer:
        log_q0, log_q1 = self.access2q[gate_id]
        phys_q0 = self.temp_mapping_dict[log_q0]
        phys_q1 = self.temp_mapping_dict[log_q1]
        
        dist = self.distance_matrix[phys_q0][phys_q1]
        criticality = self.dag_dependencies_count[gate_id]
        
        # Inverse distance weighted by criticality (add 1 to avoid div by zero)
        # Higher criticality gates contribute more to the harmonic sum
        weight = 1.0 + criticality * 0.1
        
        if dist == 0:
            # Gate is executable - reward this strongly
            front_layer_harmonic += weight * 10.0
        else:
            # Harmonic contribution: weight / distance
            front_layer_harmonic += weight / dist
        
        front_layer_count += weight
    
    # Normalize front layer harmonic (invert because we want lower = better)
    if front_layer_harmonic > 0:
        front_score = front_layer_count / front_layer_harmonic
    else:
        front_score = float('inf')
    
    # Extended layer: compute resonance with exponential depth decay
    extended_harmonic = 0.0
    extended_weight_sum = 0.0
    
    for gate_id in self.extended_layer:
        if gate_id in self.front_layer:
            continue
            
        log_q0, log_q1 = self.access2q[gate_id]
        phys_q0 = self.temp_mapping_dict[log_q0]
        phys_q1 = self.temp_mapping_dict[log_q1]
        
        dist = self.distance_matrix[phys_q0][phys_q1]
        depth = self.extended_layer_index.get(gate_id, 1)
        criticality = self.dag_dependencies_count[gate_id]
        
        # Exponential decay based on depth, boosted by criticality
        depth_decay = 0.7 ** depth
        crit_boost = 1.0 + 0.05 * criticality
        weight = depth_decay * crit_boost
        
        if dist == 0:
            extended_harmonic += weight * 5.0
        else:
            extended_harmonic += weight / dist
        
        extended_weight_sum += weight
    
    # Normalize extended layer (invert for minimization)
    if extended_harmonic > 0:
        extended_score = extended_weight_sum / extended_harmonic
    else:
        extended_score = 0.0
    
    # Coupling penalty: penalize if SWAP qubits are not involved in nearby gates
    # This prevents "wasted" SWAPs that move uninvolved qubits
    swap_involvement = 0.0
    for gate_id in self.front_layer:
        log_q0, log_q1 = self.access2q[gate_id]
        phys_q0_before = self.temp_mapping_dict[log_q0]
        phys_q1_before = self.temp_mapping_dict[log_q1]
        
        if p0 in (phys_q0_before, phys_q1_before) or p1 in (phys_q0_before, phys_q1_before):
            swap_involvement += 1.0
    
    involvement_penalty = 1.0 / (1.0 + swap_involvement)
    
    # Final cost: weighted combination
    cost = (3.0 * front_score + 
            1.0 * extended_score + 
            0.5 * involvement_penalty + 
            0.1 * decay_cost)
    
    return cost