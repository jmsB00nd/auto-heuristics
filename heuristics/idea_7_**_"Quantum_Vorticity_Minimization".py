# Strategy: ** "Quantum Vorticity Minimization"
# Intuition: ** In fluid dynamics, vorticity measures local rotation/circulation. When qubits need to "flow" toward their targets, we want laminar (smooth) flow, not turbulent (rotational) flow. A SWAP that reduces the total "curl" of the displacement field—where logical qubits trying to reach physical locations don't create crossing paths—should lead to fewer overall SWAPs. We measure this by computing a discrete circulation metric around the swap site.

**
# Stats: {'mean_swaps': 598.0555555555555, 'mean_depth': 964.1111111111111, 'mean_runtime': 1.4388263887829251, 'total_circuits': 90, 'successful_runs': 90, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    p0, p1 = swap_gate
    
    cost = 0.0
    
    # Compute displacement vectors for front layer gates (highest priority)
    front_layer_cost = 0.0
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1_curr = self.temp_mapping_dict[q1_log]
        p2_curr = self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[p1_curr][p2_curr]
        criticality = self.dag_dependencies_count[gate_id]
        
        # Weight by criticality - gates blocking more future gates matter more
        front_layer_cost += dist * (1.0 + 0.1 * criticality)
    
    cost += front_layer_cost * 4.0  # Front layer has highest weight
    
    # Vorticity calculation: measure path crossing in extended layer
    # Build displacement field: for each logical qubit, where does it want to go?
    displacements = {}  # logical_q -> (current_phys, target_direction)
    
    for gate_id in self.extended_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1_loc = self.temp_mapping_dict[q1_log]
        p2_loc = self.temp_mapping_dict[q2_log]
        
        # Each qubit "wants" to move toward the other
        displacements[q1_log] = (p1_loc, p2_loc)
        displacements[q2_log] = (p2_loc, p1_loc)
    
    # Compute circulation/vorticity: count "crossing" displacement pairs
    # near the swap site - penalize swaps that increase local turbulence
    vorticity = 0.0
    displacement_list = list(displacements.items())
    
    for i, (q1, (src1, tgt1)) in enumerate(displacement_list):
        # Check if this qubit is near the swap site
        near_swap = (self.distance_matrix[src1][p0] <= 2 or 
                     self.distance_matrix[src1][p1] <= 2)
        
        if not near_swap:
            continue
            
        for j in range(i + 1, len(displacement_list)):
            q2, (src2, tgt2) = displacement_list[j]
            
            # Detect crossing: q1 wants to go where q2 is, and vice versa
            # This creates rotational "turbulence" requiring extra swaps
            cross_score = 0.0
            if self.distance_matrix[src1][tgt2] < self.distance_matrix[src1][tgt1]:
                if self.distance_matrix[src2][tgt1] < self.distance_matrix[src2][tgt2]:
                    # Paths are crossing - bad vorticity
                    cross_score = 1.0
            
            vorticity += cross_score
    
    cost += vorticity * 0.5
    
    # Extended layer distance with depth decay
    extended_cost = 0.0
    for gate_id in self.extended_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1_curr = self.temp_mapping_dict[q1_log]
        p2_curr = self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[p1_curr][p2_curr]
        depth = self.extended_layer_index.get(gate_id, 1)
        
        # Exponential decay with depth - closer gates matter more
        weight = 1.0 / (1.0 + depth)
        extended_cost += dist * weight
    
    cost += extended_cost
    
    # Decay penalty: avoid overusing "hot" qubits
    decay_penalty = (self.decay_parameter[p0] + self.decay_parameter[p1])
    cost += decay_penalty * 0.5
    
    return cost