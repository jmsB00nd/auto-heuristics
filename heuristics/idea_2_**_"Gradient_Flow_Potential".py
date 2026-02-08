# Strategy: ** "Gradient Flow Potential"
# Intuition: ** Model the routing problem as a potential field where each gate creates an "attractive force" proportional to its criticality, but the cost is based on the *change in potential energy* (gradient) rather than absolute distance. A good SWAP should create a steep negative gradient—meaning it moves qubits "downhill" toward their targets faster than alternatives. We penalize SWAPs that create positive gradients (moving away) or shallow gradients (inefficient progress).

**
# Stats: {'mean_swaps': 807.5454545454545, 'mean_depth': 1007.5909090909091, 'mean_runtime': 4.952936952764338, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    """
    Gradient Flow Potential Heuristic
    
    Models routing as potential field minimization where we evaluate
    the "flow" (rate of distance reduction) created by a SWAP.
    """
    import math
    
    phys_a, phys_b = swap_gate
    
    # Compute decay factor - prefer swapping on "cooler" qubits
    decay_a = self.decay_parameter[phys_a]
    decay_b = self.decay_parameter[phys_b]
    thermal_factor = math.sqrt(decay_a * decay_b)  # Geometric mean
    
    # Build reverse mapping: physical -> logical (before swap was applied)
    # Since temp_mapping_dict is POST-swap, we need to reason about the delta
    
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    
    if front_layer_size == 0:
        return 0.0
    
    # === FRONT LAYER: Gradient computation ===
    # For each gate, compute how much "potential" remains (distance * criticality)
    # Lower is better since we minimize
    
    front_potential = 0.0
    front_gradient_quality = 0.0
    
    total_front_criticality = 0
    for g in self.front_layer:
        total_front_criticality += self.dag_dependencies_count[g] + 1
    
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        P1 = self.temp_mapping_dict[q1]
        P2 = self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[P1][P2]
        criticality = self.dag_dependencies_count[g] + 1
        
        # Normalized criticality weight
        weight = criticality / total_front_criticality
        
        # Potential contribution: weighted by criticality^2 to heavily favor critical gates
        front_potential += (criticality ** 1.5) * dist
        
        # Gradient quality: reward configurations where distance is small
        # Use inverse square to create strong attraction when close
        if dist > 0:
            front_gradient_quality += weight * (dist ** 2)
        # dist == 0 means gate is executable, best case
    
    # === EXTENDED LAYER: Decaying lookahead potential ===
    extended_potential = 0.0
    
    if extended_layer_size > 0:
        # Compute max depth for normalization
        max_depth = max(self.extended_layer_index.get(g, 0) for g in self.extended_layer) + 1
        
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            P1 = self.temp_mapping_dict[q1]
            P2 = self.temp_mapping_dict[q2]
            
            dist = self.distance_matrix[P1][P2]
            depth = self.extended_layer_index.get(g, 0) + 1
            criticality = self.dag_dependencies_count[g] + 1
            
            # Exponential decay by depth - distant future matters less
            depth_decay = math.exp(-0.5 * depth / max_depth)
            
            # Logarithmic criticality scaling for extended (less aggressive than front)
            crit_factor = math.log1p(criticality)
            
            extended_potential += depth_decay * crit_factor * dist
    
    # === CONGESTION PENALTY ===
    # Penalize swaps in "crowded" regions where many target qubits converge
    congestion = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        P1 = self.temp_mapping_dict[q1]
        P2 = self.temp_mapping_dict[q2]
        
        # If swap qubits are on the path of many gates, congestion increases
        if self.distance_matrix[phys_a][P1] + self.distance_matrix[phys_a][P2] <= \
           self.distance_matrix[P1][P2] + 1:
            congestion += 0.1
        if self.distance_matrix[phys_b][P1] + self.distance_matrix[phys_b][P2] <= \
           self.distance_matrix[P1][P2] + 1:
            congestion += 0.1
    
    # === COMBINE INTO FINAL COST ===
    # Front layer dominates, extended provides tie-breaking
    W_front = 1.0
    W_extended = 0.3
    W_gradient = 0.5
    W_congestion = 0.2
    
    H = thermal_factor * (
        W_front * (front_potential / front_layer_size) +
        W_extended * (extended_potential / max(extended_layer_size, 1)) +
        W_gradient * front_gradient_quality -
        W_congestion * congestion  # Congestion is subtracted (good to be on the path)
    )
    
    return H