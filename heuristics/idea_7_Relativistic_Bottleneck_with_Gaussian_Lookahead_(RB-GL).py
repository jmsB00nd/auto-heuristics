# Strategy: Relativistic Bottleneck with Gaussian Lookahead (RB-GL)
# Intuition: RB-GL differentiates between immediate "high-tension" bottlenecks and future "low-tension" signals. It applies a super-linear distance penalty ($d^{1.4}$) to the front layer to aggressively clear ready gates, while using a Gaussian temporal filter ($e^{-depth^2/8}$) and sub-linear distance ($d^{0.6}$) for lookahead to provide stable, long-term guidance. The cost is further modulated by a "Local Viscosity" factor based on qubit decay parameters to ensure balanced utilization across the QPU hardware.
# Stats: {'mean_swaps': 526.3181818181819, 'mean_depth': 944.2727272727273, 'mean_runtime': 2.245372035286643, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # 1. Global Heat Penalty (Tie-breaking and anti-cycling)
    # Penalizes the action of swapping physical qubits that have been recently used.
    p1, p2 = swap_gate
    global_heat = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # 2. Front Layer: Relativistic Tension (Super-Linear)
    # Focuses on resolving the most immediate bottlenecks in the dependency graph.
    f_total, f_count = 0.0, 0
    for g_id in self.front_layer:
        qubits = self.access2q[g_id]
        if len(qubits) < 2: continue
        q1, q2 = qubits
        # Mapping state AFTER candidate swap is applied
        P1, P2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[P1][P2]
        
        # Criticality: Square root provides a stable yet strong priority for deep dependency paths
        crit = math.sqrt(self.dag_dependencies_count[g_id] + 1.0)
        # Local Viscosity: Penalizes mapping logical qubits into "hot" physical regions
        viscosity = (self.decay_parameter[P1] + self.decay_parameter[P2]) / 2.0
        
        f_total += crit * (dist ** 1.4) * viscosity
        f_count += 1
    
    # Normalized front layer cost (lower is better)
    h_f = (f_total / f_count) if f_count > 0 else 0.0

    # 3. Extended Layer: Gaussian-Gated Lookahead (Sub-Linear)
    # Future gates provide a smooth guiding field without over-steering or greedy oscillations.
    e_total, e_count = 0.0, 0
    for g_id in self.extended_layer:
        qubits = self.access2q[g_id]
        if len(qubits) < 2: continue
        q1, q2 = qubits
        P1, P2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[P1][P2]
        
        depth = self.extended_layer_index.get(g_id, 0)
        crit = math.sqrt(self.dag_dependencies_count[g_id] + 1.0)
        
        # Gaussian temporal filter (sigma=2.0) focuses on the immediate planning horizon
        time_decay = math.exp(-(depth**2) / 8.0)
        viscosity = (self.decay_parameter[P1] + self.decay_parameter[P2]) / 2.0
        
        # Sub-linear distance penalty for lookahead to maintain a smooth gradient
        e_total += crit * (dist ** 0.6) * time_decay * viscosity
        e_count += 1
    
    h_e = (e_total / e_count) if e_count > 0 else 0.0

    # 4. Final Aggregation
    # W=0.75 balances immediate needs with future alignment
    W = 0.75
    return global_heat * (h_f + W * h_e)