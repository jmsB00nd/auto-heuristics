# Strategy: "Logarithmic Criticality Decay with Qubit Affinity"
# Intuition: Gates with high dependency counts should have diminishing returns on priority (logarithmic scaling prevents outliers from dominating), while we penalize swaps that move qubits away from their "natural affinity" - the centroid of all physical locations they need to interact with in the near future.
# Stats: {'mean_swaps': 1015.7272727272727, 'mean_depth': 1271.5454545454545, 'mean_runtime': 2.315634109757163, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    # Compute affinity centroid for each logical qubit involved in pending gates
    # Affinity = average physical location of all qubits this qubit needs to interact with
    qubit_affinity_targets = {}  # logical_qubit -> list of physical targets
    
    all_gates = list(self.front_layer) + list(self.extended_layer)
    
    for g in all_gates:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        if q1 not in qubit_affinity_targets:
            qubit_affinity_targets[q1] = []
        if q2 not in qubit_affinity_targets:
            qubit_affinity_targets[q2] = []
        
        qubit_affinity_targets[q1].append(Q2)
        qubit_affinity_targets[q2].append(Q1)
    
    # Compute affinity displacement cost for the swap
    # How much does this swap move qubits away from where they need to be?
    affinity_cost = 0.0
    p1, p2 = swap_gate[0], swap_gate[1]
    
    # Find which logical qubits are at p1 and p2
    log_at_p1 = None
    log_at_p2 = None
    for log_q, phys_q in enumerate(self.temp_mapping_dict):
        if phys_q == p1:
            log_at_p2 = log_q  # After swap, this logical qubit is at p2
        elif phys_q == p2:
            log_at_p1 = log_q  # After swap, this logical qubit is at p1
    
    for log_q, new_phys in [(log_at_p1, p1), (log_at_p2, p2)]:
        if log_q is not None and log_q in qubit_affinity_targets:
            targets = qubit_affinity_targets[log_q]
            if targets:
                avg_dist = sum(self.distance_matrix[new_phys][t] for t in targets) / len(targets)
                affinity_cost += avg_dist
    
    # Front layer: logarithmic criticality weighting
    front_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        deps = self.dag_dependencies_count[g]
        log_weight = math.log2(deps + 2)  # +2 to avoid log(1)=0
        
        front_cost += log_weight * self.distance_matrix[Q1][Q2]
    
    # Extended layer: depth-dampened logarithmic criticality
    extended_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        depth = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        log_weight = math.log2(deps + 2)
        
        # Exponential decay with depth
        depth_factor = 0.7 ** depth
        extended_cost += log_weight * depth_factor * self.distance_matrix[Q1][Q2]
    
    # Decay penalty: prefer swaps on cooler qubits
    decay_penalty = self.decay_parameter[p1] * self.decay_parameter[p2]
    
    # Normalize components
    front_size = len(self.front_layer) if self.front_layer else 1
    extended_size = len(self.extended_layer) if self.extended_layer else 1
    
    H = decay_penalty * (
        (front_cost / front_size) + 
        0.5 * (extended_cost / extended_size) +
        0.3 * affinity_cost
    )
    
    return H