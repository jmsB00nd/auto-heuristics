# Strategy: Multi-Phase Elastic Interaction (MPEI)
# Intuition: This function applies high-stiffness tension (exponent 1.5) to the front layer to aggressively resolve immediate bottlenecks, while using sub-linear lookahead tension (exponent 0.5) to provide a smooth pull for future gates that prevents greedy oscillations.
# Stats: {'mean_swaps': 539.0909090909091, 'mean_depth': 925.2272727272727, 'mean_runtime': 2.0275197462602095, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Multi-Phase Elastic Interaction (MPEI)
    # The front layer acts as a high-tension spring (p=1.5) to prioritize active gates.
    # The extended layer acts as a low-tension "magnetic pull" (p=0.5) for future alignment.
    
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    
    # Qubit usage penalty to prevent local thrashing
    q1_phys, q2_phys = swap_gate
    decay_factor = max(self.decay_parameter[q1_phys], self.decay_parameter[q2_phys])
    
    f_energy = 0.0
    for g_id in self.front_layer:
        logical_qubits = self.access2q[g_id]
        if not logical_qubits: continue
        lq1, lq2 = logical_qubits
        pq1, pq2 = self.temp_mapping_dict[lq1], self.temp_mapping_dict[lq2]
        
        dist = self.distance_matrix[pq1][pq2]
        criticality = self.dag_dependencies_count[g_id] + 1
        # Super-linear penalty for immediate gates
        f_energy += criticality * (dist ** 1.5)
        
    e_energy = 0.0
    for g_id in self.extended_layer:
        logical_qubits = self.access2q[g_id]
        if not logical_qubits: continue
        lq1, lq2 = logical_qubits
        pq1, pq2 = self.temp_mapping_dict[lq1], self.temp_mapping_dict[lq2]
        
        dist = self.distance_matrix[pq1][pq2]
        depth_factor = self.extended_layer_index.get(g_id, 0) + 1
        criticality = self.dag_dependencies_count[g_id] + 1
        # Sub-linear pull for future gates to provide a smooth gradient
        e_energy += (criticality * (dist ** 0.5)) / depth_factor

    # Weigh lookahead at 40% of the front layer's average impact
    W = 0.4
    h_front = (f_energy / front_layer_size) if front_layer_size > 0 else 0
    h_ext = (e_energy / extended_layer_size) if extended_layer_size > 0 else 0
    
    return decay_factor * (h_front + W * h_ext)