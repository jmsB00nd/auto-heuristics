# Strategy: **Demand-Asymmetric Momentum Transfer**
# Intuition: A SWAP physically exchanges two logical qubits' positions. If one logical qubit has high future "demand" (appears in many upcoming gates) while the other has low demand, we should prefer SWAPs that move the high-demand qubit closer to its interaction partners while the low-demand qubit absorbs the displacement cost. This treats the SWAP as a momentum transfer where we want to accelerate high-demand qubits toward their targets.
# Stats: {'mean_swaps': 13549.866666666667, 'mean_depth': 10616.31111111111, 'mean_runtime': 14.312183883455065, 'total_circuits': 90, 'successful_runs': 90, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    p0, p1 = swap_gate
    
    # Find which logical qubits occupy these physical positions (after swap applied)
    log_at_p0 = None
    log_at_p1 = None
    for log_q, phys_q in enumerate(self.temp_mapping_dict):
        if phys_q == p0:
            log_at_p0 = log_q
        elif phys_q == p1:
            log_at_p1 = log_q
    
    # Compute demand for each logical qubit in lookahead window
    demand = {}
    partner_targets = {}  # physical locations of interaction partners
    
    all_gates = list(self.front_layer) + list(self.extended_layer)
    
    for gate_id in all_gates:
        q1_log, q2_log = self.access2q[gate_id]
        depth = self.extended_layer_index.get(gate_id, 0)
        weight = 1.0 / (1.0 + depth)
        
        # Accumulate demand
        demand[q1_log] = demand.get(q1_log, 0) + weight
        demand[q2_log] = demand.get(q2_log, 0) + weight
        
        # Track partner locations for each logical qubit
        if q1_log not in partner_targets:
            partner_targets[q1_log] = []
        if q2_log not in partner_targets:
            partner_targets[q2_log] = []
        
        partner_targets[q1_log].append((self.temp_mapping_dict[q2_log], weight))
        partner_targets[q2_log].append((self.temp_mapping_dict[q1_log], weight))
    
    # Compute weighted distance to partners for each logical qubit at its position
    def weighted_partner_distance(log_q, phys_pos):
        if log_q is None or log_q not in partner_targets:
            return 0
        total = 0
        for partner_phys, w in partner_targets[log_q]:
            total += w * self.distance_matrix[phys_pos][partner_phys]
        return total
    
    # Current positions after swap
    dist_log0 = weighted_partner_distance(log_at_p0, p0)
    dist_log1 = weighted_partner_distance(log_at_p1, p1)
    
    # Get demands (higher demand = more important to minimize distance)
    demand_0 = demand.get(log_at_p0, 0) if log_at_p0 is not None else 0
    demand_1 = demand.get(log_at_p1, 0) if log_at_p1 is not None else 0
    
    # Demand-weighted distance: penalize high-demand qubits being far from partners
    momentum_cost = (1 + demand_0) * dist_log0 + (1 + demand_1) * dist_log1
    
    # Front layer priority: immediate gates get extra attention
    front_cost = 0
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p_q1 = self.temp_mapping_dict[q1_log]
        p_q2 = self.temp_mapping_dict[q2_log]
        front_cost += self.distance_matrix[p_q1][p_q2]
    
    # Decay penalty to avoid overusing qubits
    decay_penalty = self.decay_parameter[p0] + self.decay_parameter[p1]
    
    # Criticality bonus: prefer swaps that help gates blocking many others
    criticality_bonus = 0
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        if log_at_p0 in (q1_log, q2_log) or log_at_p1 in (q1_log, q2_log):
            criticality_bonus += self.dag_dependencies_count[gate_id] * 0.01
    
    cost = front_cost * 3.0 + momentum_cost * 0.5 + decay_penalty * 5.0 - criticality_bonus
    
    return cost