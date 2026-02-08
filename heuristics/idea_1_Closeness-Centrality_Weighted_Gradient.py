# Strategy: Closeness-Centrality Weighted Gradient
# Intuition: Standard distance heuristics treat all positions equally, but physical qubits in the center of the coupling graph (high Closeness Centrality) allow for faster global connectivity. This heuristic weights the distance cost by the "Farness" (sum of distances to all other nodes) of the target physical qubits, creating a centripetal gradient that naturally steers active logical qubits toward the chip's core to minimize future routing overhead.
# Stats: {'mean_swaps': 654.7727272727273, 'mean_depth': 983.4090909090909, 'mean_runtime': 9.747107169844888, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Helper to calculate 'Farness' (inverse of Closeness Centrality)
    # Farness(Q) = Sum of distances from Q to all other physical qubits.
    # Central nodes have low Farness; peripheral nodes have high Farness.
    # We compute this on-the-fly as it's O(N) per qubit and N is small (~100).
    def get_farness(phys_q):
        return sum(self.distance_matrix[phys_q])

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    # --- Front Layer ---
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        deps = self.dag_dependencies_count[g]
        
        # Calculate average farness of the pair relative to the system size
        # This acts as a penalty multiplier:
        # - Center: Low multiplier (preserves score)
        # - Edge: High multiplier (penalizes score)
        avg_farness = (get_farness(Q1) + get_farness(Q2)) / (2.0 * self.num_qubits)
        
        # Cost = Criticality * Distance * Farness_Penalty
        f_cost += (deps + 1) * dist * avg_farness

    front_layer_size = len(self.front_layer)
    f_norm = f_cost / front_layer_size if front_layer_size else 0

    # --- Extended Layer ---
    e_cost = 0.0
    extended_layer_size = len(self.extended_layer)
    if extended_layer_size:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            
            dist = self.distance_matrix[Q1][Q2]
            deps = self.dag_dependencies_count[g]
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            
            avg_farness = (get_farness(Q1) + get_farness(Q2)) / (2.0 * self.num_qubits)
            
            # Apply farness penalty to lookahead to guide global structure
            e_cost += (deps + 1) * dist * avg_farness * (1.0 / layer_factor)
        
        e_norm = e_cost / extended_layer_size
    else:
        e_norm = 0

    W = 0.5
    H = max_decay * (f_norm + W * e_norm)

    return H