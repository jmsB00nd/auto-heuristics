# Strategy: "Log-Saliency Quadratic Tension" (LSQT)
# Intuition: This function prioritizes immediate gates by applying a quadratic penalty to their distance, forcing the router to resolve large qubit separations aggressively. By using a logarithmic transformation on gate criticality, it prevents extremely deep dependency chains from overshadowing other parallel operations, while a harmonic decay for the extended layer ensures future gates provide guidance without causing local oscillation.
# Stats: {'mean_swaps': 540.3636363636364, 'mean_depth': 980.6363636363636, 'mean_runtime': 2.390148021958091, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    cost = 0.0
    
    # 1. Evaluate Front Layer (Immediate execution priority)
    # We use a quadratic distance penalty to heavily discourage large separations
    # for gates that are ready to execute.
    for gate_id in self.front_layer:
        # Get logical qubits and their hypothetical physical locations after the swap
        q_logs = self.access2q[gate_id]
        p1, p2 = self.temp_mapping_dict[q_logs[0]], self.temp_mapping_dict[q_logs[1]]
        
        # Physical distance on the coupling graph
        dist = self.distance_matrix[p1][p2]
        
        # Criticality: size of the transitive closure (how many gates this one blocks)
        crit = self.dag_dependencies_count[gate_id]
        
        # Logarithmic scaling for saliency prevents single deep paths from dominating
        # while Quadratic distance forces resolution of long-range conflicts.
        saliency = math.log(crit + 2.0)
        cost += saliency * (dist ** 2)

    # 2. Evaluate Extended Layer (Lookahead / Future priority)
    # Future gates use linear distance and a harmonic depth decay.
    for gate_id in self.extended_layer:
        q_logs = self.access2q[gate_id]
        p1, p2 = self.temp_mapping_dict[q_logs[0]], self.temp_mapping_dict[q_logs[1]]
        
        dist = self.distance_matrix[p1][p2]
        crit = self.dag_dependencies_count[gate_id]
        depth = self.extended_layer_index[gate_id]
        
        # Lookahead weight decays as the gate appears further in the future
        # depth=0 is the immediate successor.
        lookahead_factor = 1.0 / (depth + 2.0)
        saliency = math.log(crit + 2.0)
        cost += lookahead_factor * saliency * dist

    # 3. Qubit Heat / Congestion Penalty
    # We incorporate the decay_parameter (heat) to break ties and prevent
    # the router from thrashing (repeatedly swapping the same qubits).
    p_q1, p_q2 = swap_gate
    heat_penalty = (self.decay_parameter[p_q1] + self.decay_parameter[p_q2]) * 10.0
    
    return cost + heat_penalty