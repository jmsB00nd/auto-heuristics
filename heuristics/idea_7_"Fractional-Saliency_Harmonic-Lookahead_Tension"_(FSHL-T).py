# Strategy: "Fractional-Saliency Harmonic-Lookahead Tension" (FSHL-T)
# Intuition: This heuristic employs a sub-linear power scaling (0.75) on gate criticality to prioritize high-impact gates without allowing them to mathematically overwhelm the lookahead. It couples this with a super-linear distance penalty (1.4) and a squared-harmonic decay for future layers, creating a balanced "tension" that pulls critical qubits together while remaining sensitive to immediate routing opportunities.
# Stats: {'mean_swaps': 554.2727272727273, 'mean_depth': 969.2727272727273, 'mean_runtime': 5.45701294595545, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Initialize cost
    total_cost = 0.0
    
    # 1. Feature Engineering: Define saliency and distance metrics
    # We use a fractional power (0.75) for criticality to damp outliers 
    # and a fractional power (1.4) for distance to punish separation 
    # more than linear but less than quadratic.
    
    # Process Front Layer (Highest Priority)
    for gate_id in self.front_layer:
        # Access logical qubits and their mapped physical locations
        l_qubits = self.access2q[gate_id]
        if len(l_qubits) < 2:
            continue
        lq1, lq2 = l_qubits[0], l_qubits[1]
        
        p1 = self.temp_mapping_dict[lq1]
        p2 = self.temp_mapping_dict[lq2]
        
        dist = self.distance_matrix[p1][p2]
        crit = self.dag_dependencies_count[gate_id]
        
        # Calculate local tension
        saliency = (float(crit) + 1.1) ** 0.75
        total_cost += saliency * (float(dist) ** 1.4)
        
    # 2. Process Extended Layer (Lookahead Window)
    for gate_id in self.extended_layer:
        l_qubits = self.access2q[gate_id]
        if len(l_qubits) < 2:
            continue
        lq1, lq2 = l_qubits[0], l_qubits[1]
        
        p1 = self.temp_mapping_dict[lq1]
        p2 = self.temp_mapping_dict[lq2]
        
        dist = self.distance_matrix[p1][p2]
        depth = self.extended_layer_index[gate_id]
        crit = self.dag_dependencies_count[gate_id]
        
        # Lookahead Weight: Harmonic Squared Decay (1/1, 1/4, 1/9...)
        # depth 0 in extended layer is the immediate successor
        lookahead_weight = 1.0 / ((float(depth) + 1.0) ** 2.0)
        
        saliency = (float(crit) + 1.1) ** 0.75
        total_cost += lookahead_weight * saliency * (float(dist) ** 1.4)
        
    # 3. Qubit Health (Heat) Penalty
    # Acts as a tie-breaker to prevent repeated use of the same physical links
    heat_score = self.decay_parameter[swap_gate[0]] + self.decay_parameter[swap_gate[1]]
    total_cost += heat_score * 50.0
    
    return float(total_cost)