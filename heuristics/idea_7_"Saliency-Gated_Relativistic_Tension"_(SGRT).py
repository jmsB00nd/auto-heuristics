# Strategy: "Saliency-Gated Relativistic Tension" (SGRT)
# Intuition: This function treats the quantum circuit as a multi-layered elastic system where high-criticality gates exert a logarithmic "pull" on logical qubits. It uses differentiated polynomial exponents for immediate and future layers to aggressively resolve current bottlenecks while maintaining a "relativistic" focus on distant future dependencies.
# Stats: {'mean_swaps': 541.2727272727273, 'mean_depth': 972.8181818181819, 'mean_runtime': 1.019491520794955, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    # 1. Physical Qubit Heat Penalty
    # Bias against physical qubits that have high decay or noise levels.
    # This prevents the router from thrashing (repeated swaps) and minimizes noise.
    q_u, q_v = swap_gate
    heat_bias = (self.decay_parameter[q_u] + self.decay_parameter[q_v]) * 2.5
    
    total_tension = 0.0
    
    # 2. Front Layer Tension (Immediate Urgency)
    # We apply a higher polynomial exponent (1.4) to distance to prioritize 
    # bringing logical qubits together for gates ready to execute NOW.
    for gate_id in self.front_layer:
        log_qubits = self.access2q[gate_id]
        if len(log_qubits) == 2:
            lq1, lq2 = log_qubits
            p1 = self.temp_mapping_dict[lq1]
            p2 = self.temp_mapping_dict[lq2]
            dist = self.distance_matrix[p1][p2]
            
            # Saliency is logarithmic to the transitive closure (criticality)
            # This ensures important gates pull harder but don't overwhelm the heuristic.
            saliency = math.log(self.dag_dependencies_count[gate_id] + 2.0)
            total_tension += saliency * (dist ** 1.4)
            
    # 3. Extended Layer Tension (Lookahead Horizon)
    # Future gates exert pull that decays with depth. We use a softer distance 
    # exponent (1.1) and a sub-linear depth decay to keep the lookahead relevant.
    for gate_id in self.extended_layer:
        log_qubits = self.access2q[gate_id]
        if len(log_qubits) == 2:
            lq1, lq2 = log_qubits
            p1 = self.temp_mapping_dict[lq1]
            p2 = self.temp_mapping_dict[lq2]
            dist = self.distance_matrix[p1][p2]
            
            depth = self.extended_layer_index[gate_id]
            saliency = math.log(self.dag_dependencies_count[gate_id] + 2.0)
            
            # Depth decay factor (depth+1.5)^0.75 balances immediate vs future needs.
            decay_factor = (depth + 1.5) ** 0.75
            weight = saliency / decay_factor
            total_tension += weight * (dist ** 1.1)
            
    return float(total_tension + heat_bias)