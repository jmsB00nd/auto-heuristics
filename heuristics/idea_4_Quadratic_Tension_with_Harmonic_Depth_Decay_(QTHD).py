# Strategy: Quadratic Tension with Harmonic Depth Decay (QTHD)
# Intuition: By squaring the distances in the front layer, we create a high-tension "elastic" cost that aggressively penalizes the most separated critical gates, while using an inverse-square depth decay for the lookahead to ensure future gates provide guidance without distracting from immediate bottlenecks.
# Stats: {'mean_swaps': 512.3181818181819, 'mean_depth': 911.4090909090909, 'mean_runtime': 2.2666232477534902, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    f_size = len(self.front_layer)
    e_size = len(self.extended_layer)
    
    # 1. Front Layer: Quadratic Distance Scaling
    # Squaring the distance creates a "potential energy" effect that forces 
    # the router to prioritize resolving the most distant immediate gates first.
    f_cost = 0
    for g in self.front_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        q1, q2 = qs
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        crit = self.dag_dependencies_count[g]
        
        # Quadratic penalty for distance weighted by closure size
        f_cost += (crit + 1) * (self.distance_matrix[Q1][Q2] ** 2)

    # 2. Extended Layer: Inverse-Square Depth Weighting
    # Future gates provide a "gravitational pull," but their influence 
    # drops off quadratically with their depth in the lookahead window.
    e_cost = 0
    for g in self.extended_layer:
        qs = self.access2q[g]
        if len(qs) < 2: continue
        q1, q2 = qs
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        depth = self.extended_layer_index.get(g, 0) + 1
        crit = self.dag_dependencies_count[g]
        
        # Linear distance for lookahead, but hyperbolic decay of importance
        e_cost += ((crit + 1) * self.distance_matrix[Q1][Q2]) / (depth ** 2)

    # 3. Normalization and Qubit Heat Penalty
    # We use the max decay of the physical qubits involved in the candidate SWAP
    # to avoid moving logical qubits onto highly congested physical nodes.
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    h_front = f_cost / f_size if f_size > 0 else 0
    h_ext = e_cost / e_size if e_size > 0 else 0
    
    # Total heuristic: Balanced sum scaled by physical qubit usage (heat)
    return max_decay * (h_front + h_ext)