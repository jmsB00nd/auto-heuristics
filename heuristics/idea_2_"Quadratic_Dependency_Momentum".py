# Strategy: "Quadratic Dependency Momentum"
# Intuition: Gates with high dependency counts create cascading delays - the cost of keeping them blocked grows quadratically with their criticality. Additionally, we should penalize swaps that move qubits away from their "center of gravity" (the average position of all qubits they need to interact with in the extended layer).
# Stats: {'mean_swaps': 563.7272727272727, 'mean_depth': 930.5909090909091, 'mean_runtime': 4.365999611941251, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    p1, p2 = swap_gate
    
    # Compute center of gravity for each logical qubit involved in pending gates
    # This represents where each qubit "wants to be" based on future interactions
    qubit_targets = {}  # logical_qubit -> list of physical target positions
    
    for g in self.front_layer | set(self.extended_layer):
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        if q1 not in qubit_targets:
            qubit_targets[q1] = []
        if q2 not in qubit_targets:
            qubit_targets[q2] = []
        
        qubit_targets[q1].append(Q2)  # q1 wants to be near Q2's position
        qubit_targets[q2].append(Q1)  # q2 wants to be near Q1's position
    
    # Front layer: quadratic dependency weighting
    front_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        
        # Quadratic scaling: blocking a critical gate is increasingly expensive
        criticality = (deps + 1) ** 2
        front_cost += criticality * self.distance_matrix[Q1][Q2]
    
    # Extended layer: momentum toward center of gravity
    extended_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        depth = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        
        # Diminishing weight for deeper gates, but still quadratic in criticality
        weight = ((deps + 1) ** 1.5) / (depth ** 0.5)
        extended_cost += weight * self.distance_matrix[Q1][Q2]
    
    # Center of gravity alignment bonus/penalty
    # Reward swaps that move qubits closer to their interaction partners
    cog_penalty = 0.0
    for log_q, targets in qubit_targets.items():
        if not targets:
            continue
        phys_pos = self.temp_mapping_dict[log_q]
        
        # Compute average distance to all target positions
        avg_dist = sum(self.distance_matrix[phys_pos][t] for t in targets) / len(targets)
        cog_penalty += avg_dist
    
    # Decay factor: prefer qubits with lower heat
    decay_factor = (self.decay_parameter[p1] + self.decay_parameter[p2]) / 2.0
    
    # Normalize components
    n_front = max(len(self.front_layer), 1)
    n_ext = max(len(self.extended_layer), 1)
    n_cog = max(len(qubit_targets), 1)
    
    H = decay_factor * (
        (front_cost / n_front) + 
        0.5 * (extended_cost / n_ext) + 
        0.3 * (cog_penalty / n_cog)
    )
    
    return H