# Strategy: State-Adaptive Lookahead Gating (SALG)
# Intuition: The importance of future-looking optimizations should be inversely proportional to the current "tension" in the front layer. By dynamically scaling the lookahead weight based on the average distance of immediate gates, the router focuses exclusively on urgent bottlenecks when they exist and shifts to proactive positioning only as the current layer converges.
# Stats: {'mean_swaps': 578.6363636363636, 'mean_depth': 950.8181818181819, 'mean_runtime': 2.023217710581693, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    
    # 1. Front Layer Analysis: Calculate weighted tension and convergence state
    f_weighted_tension = 0
    f_raw_dist_sum = 0
    
    for g in self.front_layer:
        # access2q returns logical qubits; for single-qubit gates it returns an empty list
        logical_qubits = self.access2q[g]
        if not logical_qubits:
            continue
            
        q1, q2 = logical_qubits
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        criticality = self.dag_dependencies_count[g]
        
        f_raw_dist_sum += dist
        # Quadratic distance penalty to aggressively prioritize resolving long-range separations
        f_weighted_tension += (criticality + 1) * (dist ** 2)
        
    f_score = f_weighted_tension / front_layer_size if front_layer_size > 0 else 0
    f_avg_dist = f_raw_dist_sum / front_layer_size if front_layer_size > 0 else 0
    
    # 2. Dynamic Lookahead Gating
    # The weight of the extended layer is gated by the convergence of the front layer.
    # As f_avg_dist approaches 0, the weight increases, allowing future gates to guide movement.
    dynamic_W = 1.0 / (1.0 + f_avg_dist)
    
    # 3. Extended Layer Analysis: Calculate future interaction pressure
    e_score = 0
    if extended_layer_size > 0:
        e_weighted_tension = 0
        for g in self.extended_layer:
            logical_qubits = self.access2q[g]
            if not logical_qubits:
                continue
                
            q1, q2 = logical_qubits
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            criticality = self.dag_dependencies_count[g]
            # Use lookahead depth to decay influence of distant future gates
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            
            e_weighted_tension += ((criticality + 1) * dist) / layer_factor
            
        e_score = e_weighted_tension / extended_layer_size
        
    # 4. Anti-Thrashing Decay
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    # 5. Final Composition
    # H = Decay * (Immediate_Score + Dynamic_Weight * Future_Score)
    cost = max_decay * (f_score + dynamic_W * e_score)
    
    return cost