# Strategy: ** "Entropic Interaction Dispersion"
# Intuition: ** A good SWAP should reduce the "disorder" in how logical qubits are spread across the physical topology. By measuring the entropy of distance distributions for pending gates, we penalize swaps that leave qubits in high-entropy (scattered) configurations while rewarding swaps that cluster interacting qubits together, weighted by their dependency criticality.

**
# Stats: {'mean_swaps': 538.8636363636364, 'mean_depth': 973.7272727272727, 'mean_runtime': 1.1493855281309648, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    # Compute distance distribution entropy for front layer
    front_distances = []
    front_criticality_sum = 0
    
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        crit = self.dag_dependencies_count[g] + 1
        front_distances.append(dist)
        front_criticality_sum += crit * dist
    
    # Compute entropy of front layer distances
    if front_distances:
        total_dist = sum(front_distances) + len(front_distances)  # +1 smoothing
        probs = [(d + 1) / total_dist for d in front_distances]
        front_entropy = -sum(p * math.log(p + 1e-10) for p in probs)
    else:
        front_entropy = 0
    
    # Extended layer with depth-decayed entropy contribution
    ext_dispersion = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        depth = self.extended_layer_index.get(g, 0) + 1
        crit = self.dag_dependencies_count[g] + 1
        
        # Inverse depth weighting with criticality
        ext_dispersion += (crit * dist * dist) / (depth * depth)
    
    # Normalize extended contribution
    ext_term = ext_dispersion / (len(self.extended_layer) + 1)
    
    # Decay factor: penalize hot qubits
    swap_heat = self.decay_parameter[swap_gate[0]] * self.decay_parameter[swap_gate[1]]
    
    # Combine: entropy measures disorder, criticality_sum measures urgency
    front_size = len(self.front_layer) if self.front_layer else 1
    
    H = swap_heat * (
        (front_criticality_sum / front_size) * (1 + front_entropy) + 
        0.5 * ext_term
    )
    
    return H