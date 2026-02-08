# Strategy: "Log-Saliency Power-Law Tension" (LS-PLT)
# Intuition: This function treats qubit distance as a "tension" field where the penalty grows as a power law (quadratic for immediate gates) to aggressively prioritize closing gaps, but uses a logarithmic "saliency" mask on gate criticality to prevent massive downstream dependency chains from monopolizing the routing logic at the expense of local parallelism.
# Stats: {'mean_swaps': 534.1818181818181, 'mean_depth': 958.6818181818181, 'mean_runtime': 1.1190438812429255, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # 1. Physical Qubit Heat Penalty
    # We incorporate the decay_parameter to avoid over-utilizing the same physical qubits/paths.
    heat_penalty = (self.decay_parameter[swap_gate[0]] + self.decay_parameter[swap_gate[1]]) * 15.0

    score = 0.0

    # 2. Front Layer Tension (Immediate Gates)
    # Uses quadratic distance penalty to force immediate alignment.
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[p1][p2]
        # Saliency: dampen criticality with log to focus on relative importance
        saliency = math.log(self.dag_dependencies_count[gate_id] + 2.0)
        
        score += (dist ** 2.0) * saliency

    # 3. Extended Layer Tension (Lookahead Gates)
    # Uses a lower power law (1.4) and a polynomial depth decay.
    # This creates a "softer" attraction for future gates.
    for gate_id in self.extended_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[p1][p2]
        crit = self.dag_dependencies_count[gate_id]
        depth = self.extended_layer_index[gate_id]
        
        # Temporal Weight: Quadratic decay with depth to focus on the near-future
        temporal_weight = 1.0 / ((depth + 1.0) ** 2.0)
        saliency = math.log(crit + 2.0)
        
        # Power-law distance (1.4) provides non-linear pull without dominating the front layer
        score += (dist ** 1.4) * saliency * temporal_weight

    return score + heat_penalty