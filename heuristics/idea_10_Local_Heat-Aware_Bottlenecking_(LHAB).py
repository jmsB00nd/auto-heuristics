# Strategy: Local Heat-Aware Bottlenecking (LHAB)
# Intuition: Standard quantum routing heuristics use a global decay multiplier to prevent infinite swap loops, but this often ignores the specific congestion or "heat" of the physical qubits being utilized in a candidate swap. LHAB introduces a gate-specific "local heat" penalty by weighting the physical distance of each gate by the recent activity of its target physical qubits. This encourages the router to find paths through "cooler" (less recently swapped) areas of the chip, effectively load-balancing the operations and preventing local bottlenecks that a global multiplier would miss.
# Stats: {'mean_swaps': 531.9090909090909, 'mean_depth': 937.0454545454545, 'mean_runtime': 4.8953318704258315, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Weight for balancing the influence of the extended lookahead layer
    W = 1.0
    
    # Global anti-cycle penalty for the candidate swap itself
    # Mimics the base heuristic's global scaling to prevent infinite swap cycles
    q_s1, q_s2 = swap_gate
    swap_decay = max(self.decay_parameter[q_s1], self.decay_parameter[q_s2])
    
    f_cost = 0
    f_count = 0
    for g in self.front_layer:
        qubits = self.access2q[g]
        if not qubits: 
            continue # Skip single-qubit gates as they don't influence routing distance
        
        q1, q2 = qubits
        # Physical locations of logical qubits after the candidate swap is applied
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        crit = self.dag_dependencies_count[g]
        
        # Local heat: measure of recent swap activity for the specific target physical qubits
        local_heat = (self.decay_parameter[Q1] + self.decay_parameter[Q2]) / 2.0
        
        # Use a quadratic distance penalty for the front layer to prioritize immediate 
        # resolution of ready gates, weighted by their transitive dependency count (criticality)
        f_cost += (crit + 1) * (dist ** 2) * (1.0 + local_heat)
        f_count += 1

    e_cost = 0
    e_count = 0
    for g in self.extended_layer:
        qubits = self.access2q[g]
        if not qubits:
            continue
            
        q1, q2 = qubits
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        crit = self.dag_dependencies_count[g]
        depth = self.extended_layer_index.get(g, 0) + 1
        
        local_heat = (self.decay_parameter[Q1] + self.decay_parameter[Q2]) / 2.0
        
        # Linear distance for lookahead with exponential decay based on DAG depth.
        # This provides broad guidance toward future gates without being overly 
        # sensitive to exact positions of qubits in the distant future.
        e_cost += (crit + 1) * dist * (0.8 ** (depth - 1)) * (1.0 + local_heat)
        e_count += 1

    # Normalize layer costs by the number of contributing gates
    h_f = (f_cost / f_count) if f_count > 0 else 0
    h_e = (e_cost / e_count) if e_count > 0 else 0
    
    # Return the product of global swap decay and the tiered layer costs
    return swap_decay * (h_f + W * h_e)