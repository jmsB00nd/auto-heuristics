# Strategy: "Routing Entropy Collapse"
# Intuition: A good SWAP should reduce uncertainty in future routing decisions by collapsing the space of necessary movements. We measure how much a SWAP "commits" the routing toward a deterministic path by computing the entropy of remaining distances for dependent gates - lower entropy after SWAP means we've made a decisive move that narrows down the optimal path.
# Stats: {'mean_swaps': 574.8666666666667, 'mean_depth': 965.3444444444444, 'mean_runtime': 1.3357157839669123, 'total_circuits': 90, 'successful_runs': 90, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    p0, p1 = swap_gate
    
    # Base cost accumulator
    total_cost = 0.0
    
    # Track routing entropy for front layer and extended layer
    front_layer_distances = []
    extended_layer_distances = []
    
    # Process front layer gates (highest priority)
    for gate_id in self.front_layer:
        log_q0, log_q1 = self.access2q[gate_id]
        phys_q0 = self.temp_mapping_dict[log_q0]
        phys_q1 = self.temp_mapping_dict[log_q1]
        
        dist = self.distance_matrix[phys_q0][phys_q1]
        front_layer_distances.append(dist)
        
        # Criticality weighting
        criticality = self.dag_dependencies_count[gate_id]
        total_cost += dist * (1.0 + math.log1p(criticality))
    
    # Process extended layer with depth decay
    for gate_id in self.extended_layer:
        log_q0, log_q1 = self.access2q[gate_id]
        phys_q0 = self.temp_mapping_dict[log_q0]
        phys_q1 = self.temp_mapping_dict[log_q1]
        
        dist = self.distance_matrix[phys_q0][phys_q1]
        depth = self.extended_layer_index.get(gate_id, 1)
        
        # Exponential depth decay
        decay_factor = 1.0 / (1.0 + depth)
        extended_layer_distances.append(dist * decay_factor)
        
        criticality = self.dag_dependencies_count[gate_id]
        total_cost += dist * decay_factor * (1.0 + 0.1 * math.log1p(criticality))
    
    # NOVEL: Compute routing entropy collapse metric
    # Entropy measures "spread" of distances - we want to minimize variance
    # A decisive SWAP should make distances more uniform (lower entropy)
    all_distances = front_layer_distances + extended_layer_distances
    
    if len(all_distances) > 1:
        # Normalize distances to form a pseudo-probability distribution
        dist_sum = sum(d + 1 for d in all_distances)  # +1 to avoid zeros
        probs = [(d + 1) / dist_sum for d in all_distances]
        
        # Shannon entropy of distance distribution
        routing_entropy = -sum(p * math.log(p + 1e-10) for p in probs)
        
        # Variance of distances (high variance = uncertain routing path)
        mean_dist = sum(all_distances) / len(all_distances)
        variance = sum((d - mean_dist) ** 2 for d in all_distances) / len(all_distances)
        
        # Entropy collapse bonus: reward SWAPs that reduce routing uncertainty
        entropy_penalty = routing_entropy * math.sqrt(variance + 1)
        total_cost += 0.5 * entropy_penalty
    
    # NOVEL: Commitment gradient - how much does this SWAP "commit" us?
    # Measure if swap qubits are involved in many future operations
    swap_involvement = 0
    for gate_id in self.front_layer | set(self.extended_layer):
        log_q0, log_q1 = self.access2q[gate_id]
        phys_q0 = self.temp_mapping_dict[log_q0]
        phys_q1 = self.temp_mapping_dict[log_q1]
        
        # Check if swapped qubits are part of this gate's routing path
        if p0 in (phys_q0, phys_q1) or p1 in (phys_q0, phys_q1):
            depth = self.extended_layer_index.get(gate_id, 0)
            swap_involvement += 1.0 / (1.0 + depth)
    
    # High involvement is good if it reduces distances, penalize otherwise
    avg_front_dist = sum(front_layer_distances) / max(len(front_layer_distances), 1)
    if avg_front_dist > 1:
        total_cost += 0.1 * swap_involvement * avg_front_dist
    else:
        total_cost -= 0.2 * swap_involvement  # Reward involvement when distances are small
    
    # Qubit decay penalty (avoid hot qubits)
    decay_penalty = self.decay_parameter[p0] + self.decay_parameter[p1]
    total_cost += decay_penalty
    
    return total_cost