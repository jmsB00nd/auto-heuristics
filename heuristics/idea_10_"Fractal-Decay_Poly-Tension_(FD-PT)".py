# Strategy: "Fractal-Decay Poly-Tension (FD-PT)"
# Intuition: This function employs a "Fractal" depth decay (exponent 0.75) to maintain a wider lookahead horizon than standard harmonic decay, combined with a biquadratic penalty for immediate gates and a 1.5-power tension for future gates to ensure a smooth but urgent optimization gradient.
# Stats: {'mean_swaps': 548.6818181818181, 'mean_depth': 967.5, 'mean_runtime': 2.096585360440341, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    cost = 0.0
    
    # 1. Immediate Urgency (Front Layer)
    # Uses a quadratic distance penalty to prioritize gates ready for execution.
    for gate_id in self.front_layer:
        lq1, lq2 = self.access2q[gate_id]
        pq1 = self.temp_mapping_dict[lq1]
        pq2 = self.temp_mapping_dict[lq2]
        dist = self.distance_matrix[pq1][pq2]
        
        # log2 scaling of the transitive closure (criticality)
        # This prevents "super-nodes" from dominating the score while keeping their rank.
        saliency = math.log2(self.dag_dependencies_count[gate_id] + 2.0)
        cost += saliency * (dist ** 2.0)
        
    # 2. Future Planning (Extended Layer)
    # Fractal decay (0.75) is slower than linear decay, keeping the router
    # aware of long-term dependency bottlenecks without being nearsighted.
    for gate_id in self.extended_layer:
        lq1, lq2 = self.access2q[gate_id]
        pq1 = self.temp_mapping_dict[lq1]
        pq2 = self.temp_mapping_dict[lq2]
        dist = self.distance_matrix[pq1][pq2]
        
        depth = self.extended_layer_index[gate_id]
        saliency = math.log2(self.dag_dependencies_count[gate_id] + 2.0)
        
        # Fractal decay: (depth + 1)^-0.75
        # Distance power: 1.5 (intermediate between linear and quadratic)
        # provides a smoother potential landscape for lookahead gates.
        depth_weight = 1.0 / ((depth + 1.0) ** 0.75)
        cost += saliency * (dist ** 1.5) * depth_weight
        
    # 3. Physical Qubit Health (Decay/Heat)
    # Adds a penalty for swapping through qubits that have been used frequently,
    # helping distribute the "heat" (gate load) across the hardware topology.
    p1, p2 = swap_gate
    heat_penalty = (self.decay_parameter[p1] + self.decay_parameter[p2]) * 8.5
    
    return cost + heat_penalty