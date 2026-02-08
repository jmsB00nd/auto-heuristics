# Strategy: "Harmonic Resonance Coupling"
# Intuition: Physical qubits that are involved in many pending gates create "resonance points" - we should favor swaps that move logical qubits toward positions where they can satisfy multiple future interactions with minimal subsequent movement, weighted by the harmonic mean of distances to all their future partners.
# Stats: {'mean_swaps': 532.9090909090909, 'mean_depth': 987.7727272727273, 'mean_runtime': 1.249527010050687, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Build a map of each logical qubit to all its future interaction partners
    logical_qubit_partners = {}
    
    # Collect all gates and their priorities
    all_gates = []
    for g in self.front_layer:
        all_gates.append((g, 1.0))  # Front layer has priority 1.0
    for g in self.extended_layer:
        depth = self.extended_layer_index.get(g, 1) + 1
        all_gates.append((g, 1.0 / depth))  # Decaying priority by depth
    
    # Build partner graph with weighted urgency
    for gate_id, priority in all_gates:
        q1, q2 = self.access2q[gate_id]
        criticality = self.dag_dependencies_count[gate_id] + 1
        weight = priority * criticality
        
        if q1 not in logical_qubit_partners:
            logical_qubit_partners[q1] = []
        if q2 not in logical_qubit_partners:
            logical_qubit_partners[q2] = []
        
        logical_qubit_partners[q1].append((q2, weight))
        logical_qubit_partners[q2].append((q1, weight))
    
    # Compute harmonic coupling score for each logical qubit
    # Harmonic mean emphasizes the closest partners (bottleneck reduction)
    def harmonic_coupling_cost(logical_q):
        if logical_q not in logical_qubit_partners:
            return 0.0
        
        partners = logical_qubit_partners[logical_q]
        if not partners:
            return 0.0
        
        phys_pos = self.temp_mapping_dict[logical_q]
        
        weighted_inverse_sum = 0.0
        total_weight = 0.0
        
        for partner_q, weight in partners:
            partner_phys = self.temp_mapping_dict[partner_q]
            dist = self.distance_matrix[phys_pos][partner_phys]
            
            # Add 1 to avoid division by zero (adjacent qubits have dist=1)
            weighted_inverse_sum += weight / (dist + 1)
            total_weight += weight
        
        if weighted_inverse_sum == 0:
            return 0.0
        
        # Inverted harmonic mean: lower is better when qubits are closer
        # We want total_weight / weighted_inverse_sum to be small
        return total_weight / weighted_inverse_sum
    
    # Calculate total harmonic coupling cost across all active logical qubits
    total_coupling_cost = 0.0
    active_qubits = set()
    
    for gate_id, _ in all_gates:
        q1, q2 = self.access2q[gate_id]
        active_qubits.add(q1)
        active_qubits.add(q2)
    
    for lq in active_qubits:
        total_coupling_cost += harmonic_coupling_cost(lq)
    
    # Penalize swaps on "hot" qubits (decay parameter)
    decay_penalty = (self.decay_parameter[swap_gate[0]] + 
                     self.decay_parameter[swap_gate[1]]) / 2.0
    
    # Front layer urgency: direct distance cost for immediate gates
    front_urgency = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        P1 = self.temp_mapping_dict[q1]
        P2 = self.temp_mapping_dict[q2]
        front_urgency += self.distance_matrix[P1][P2]
    
    # Combine: front urgency dominates, harmonic coupling guides lookahead
    alpha = 0.5  # Balance factor
    
    H = decay_penalty * (front_urgency + alpha * total_coupling_cost)
    
    return H