# Strategy: "Inverse Squared Distance with Dependency Chain Propagation"
# Intuition: SWAPs that bring high-criticality gates closer should be rewarded quadratically (since routing often requires multiple hops), while penalizing moves that disrupt dependency chains. By weighting the inverse square of distances and propagating dependency importance through the extended layer exponentially, we capture both immediate benefit and downstream ripple effects.
# Stats: {'mean_swaps': 537.0454545454545, 'mean_depth': 925.0, 'mean_runtime': 1.8691079291430386, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    EPSILON = 1e-6
    FRONT_WEIGHT = 2.0
    EXTENDED_DECAY_BASE = 0.7
    
    # Compute swap "heat" penalty - prefer cooler qubits
    swap_heat = (self.decay_parameter[swap_gate[0]] + self.decay_parameter[swap_gate[1]]) / 2.0
    
    # Front layer: inverse squared distance weighted by dependency count
    front_cost = 0.0
    front_criticality_sum = 0.0
    
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        dist = self.distance_matrix[p1][p2]
        
        criticality = self.dag_dependencies_count[gate_id] + 1
        front_criticality_sum += criticality
        
        # Inverse squared weighting: closer is much better
        front_cost += criticality * (dist ** 2)
    
    # Normalize front cost
    if front_criticality_sum > EPSILON:
        front_cost = front_cost / front_criticality_sum
    
    # Extended layer: exponentially decaying importance with chain propagation
    extended_cost = 0.0
    extended_weight_sum = 0.0
    
    # Build a simple dependency chain weight for extended gates
    for gate_id in self.extended_layer:
        q1_log, q2_log = self.access2q[gate_id]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        dist = self.distance_matrix[p1][p2]
        
        depth = self.extended_layer_index.get(gate_id, 0)
        criticality = self.dag_dependencies_count[gate_id] + 1
        
        # Exponential decay based on depth
        depth_weight = EXTENDED_DECAY_BASE ** depth
        
        # Chain propagation: sqrt of criticality to dampen extreme values
        chain_weight = (criticality ** 0.5) * depth_weight
        extended_weight_sum += chain_weight
        
        # Linear distance for extended (less aggressive than front)
        extended_cost += chain_weight * dist
    
    # Normalize extended cost
    if extended_weight_sum > EPSILON:
        extended_cost = extended_cost / extended_weight_sum
    
    # Combine costs with heat penalty as multiplier
    H = swap_heat * (FRONT_WEIGHT * front_cost + extended_cost)
    
    return H