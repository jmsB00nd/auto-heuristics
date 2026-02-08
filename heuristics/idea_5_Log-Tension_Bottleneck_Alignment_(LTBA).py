# Strategy: Log-Tension Bottleneck Alignment (LTBA)
# Intuition: This cost function employs a logarithmic distance metric to favor closing small gaps (making qubits adjacent) over marginal progress on distant pairs, combined with a squared dependency weight to prioritize the primary DAG bottlenecks. By using exponential decay for the lookahead window, the router maintains a sharp focus on the immediate future while suppressing noise from distant layers.
# Stats: {'mean_swaps': 546.8636363636364, 'mean_depth': 1008.2727272727273, 'mean_runtime': 1.2035503712567417, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    # 1. Decay factor: Penalize qubits that have been recently swapped to prevent cycles/thrashing.
    # We square the max decay to aggressively increase the 'inertia' of moving qubits.
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]]) ** 2

    # 2. Front Layer Calculation (Immediate Execution Targets)
    f_tension = 0
    for g in self.front_layer:
        gate_qubits = self.access2q[g]
        if not gate_qubits: continue
        q1, q2 = gate_qubits
        
        # Current physical locations after the candidate swap
        p1, p2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[p1][p2]
        
        # Logic: Squared dependencies amplify critical path bottlenecks.
        # Logarithmic distance (log1p) ensures that reducing distance from 2 to 1 
        # is weighted more heavily than reducing it from 10 to 9.
        crit_weight = (self.dag_dependencies_count[g] + 1) ** 2
        f_tension += crit_weight * math.log1p(dist)

    # 3. Extended Layer Calculation (Lookahead Planning)
    e_tension = 0
    lookahead_decay_base = 0.5 # Exponential decay for future layers
    for g in self.extended_layer:
        gate_qubits = self.access2q[g]
        if not gate_qubits: continue
        q1, q2 = gate_qubits
        
        p1, p2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[p1][p2]
        
        # Depth-based attenuation (0-indexed depth)
        depth = self.extended_layer_index.get(g, 0)
        crit_weight = (self.dag_dependencies_count[g] + 1)
        
        # Future gates use linear criticality but exponential depth decay to filter noise.
        e_tension += (crit_weight * math.log1p(dist)) * (lookahead_decay_base ** depth)

    # Normalization by layer size to ensure stability across different circuit depths
    f_norm = f_tension / len(self.front_layer) if self.front_layer else 0
    e_norm = e_tension / len(self.extended_layer) if self.extended_layer else 0
    
    # Global lookahead weight
    W = 0.75 
    
    # Final Heuristic Value (Minimization)
    return max_decay * (f_norm + W * e_norm)