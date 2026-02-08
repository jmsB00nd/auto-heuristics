# Strategy: "Topological Flow Divergence"
# Intuition: Model the quantum circuit as a flow network where gates create "pressure" that must flow through the coupling graph. A good SWAP reduces the divergence between where qubits currently are and where they need to flow, weighted by how much "downstream traffic" each gate controls. We use a logarithmic damping to prevent extreme values while preserving ranking sensitivity.
# Stats: {'mean_swaps': 598.4333333333333, 'mean_depth': 1010.6, 'mean_runtime': 1.0759733200073243, 'total_circuits': 90, 'successful_runs': 90, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    cost = 0.0
    p0, p1 = swap_gate
    
    # Decay penalty for overused qubits (thermal pressure)
    decay_penalty = (self.decay_parameter[p0] + self.decay_parameter[p1])
    
    # Front layer: highest priority - compute flow divergence
    front_layer_cost = 0.0
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        phys_q1 = self.temp_mapping_dict[q1_log]
        phys_q2 = self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[phys_q1][phys_q2]
        
        # Criticality as flow volume
        criticality = self.dag_dependencies_count[gate_id] + 1
        
        # Flow divergence: distance * log(criticality) creates pressure gradient
        flow_divergence = dist * math.log1p(criticality)
        front_layer_cost += flow_divergence
    
    # Extended layer: compute future flow with exponential depth decay
    extended_layer_cost = 0.0
    for gate_id in self.extended_layer:
        q1_log, q2_log = self.access2q[gate_id]
        phys_q1 = self.temp_mapping_dict[q1_log]
        phys_q2 = self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[phys_q1][phys_q2]
        depth = self.extended_layer_index.get(gate_id, 1)
        criticality = self.dag_dependencies_count[gate_id] + 1
        
        # Depth-attenuated flow: further gates contribute less
        # Using 1/(depth+1)^1.5 for faster decay than linear
        attenuation = 1.0 / ((depth + 1) ** 1.5)
        
        # Combine distance with criticality using geometric mean for balance
        flow_contribution = math.sqrt(dist * math.log1p(criticality)) * attenuation
        extended_layer_cost += flow_contribution
    
    # Connectivity bonus: reward swaps that improve local graph centrality
    # Count how many front-layer qubits become adjacent after swap
    adjacency_bonus = 0.0
    front_physical = set()
    for gate_id in self.front_layer:
        q1_log, q2_log = self.access2q[gate_id]
        front_physical.add(self.temp_mapping_dict[q1_log])
        front_physical.add(self.temp_mapping_dict[q2_log])
    
    for pq in front_physical:
        # Check if swap qubits are now adjacent to needed qubits
        if self.distance_matrix[p0][pq] == 1:
            adjacency_bonus += 0.1
        if self.distance_matrix[p1][pq] == 1:
            adjacency_bonus += 0.1
    
    # Combine terms: front layer dominates, extended layer provides foresight
    # Decay penalty prevents oscillation on same qubits
    cost = (3.0 * front_layer_cost + 
            extended_layer_cost + 
            decay_penalty * 0.5 - 
            adjacency_bonus)
    
    return cost