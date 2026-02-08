# Strategy: Log-Sum-Exponential Bottleneck (LSEB)
# Intuition: Circuit execution is fundamentally limited by the "furthest" interaction in the front layer (the bottleneck). LSEB uses the Log-Sum-Exp function to provide a smooth, differentiable approximation of the maximum distance, forcing the router to prioritize resolving the most severe bottlenecks. For the extended layer, it employs a sub-linear cubic-root tension to provide stable long-term guidance without the oscillations common in quadratic or linear lookahead models.
# Stats: {'mean_swaps': 550.0454545454545, 'mean_depth': 971.0909090909091, 'mean_runtime': 1.3186111016706987, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # 1. Physical Heat Factor
    # We use the sum of decay parameters of the physical qubits involved in the swap.
    # This acts as a "viscosity" multiplier, penalizing moves through congested regions.
    p1, p2 = swap_gate[0], swap_gate[1]
    heat_multiplier = 1.0 + (self.decay_parameter[p1] + self.decay_parameter[p2])

    # 2. Front Layer: Log-Sum-Exp (LSE) Bottleneck Focus
    # Instead of an arithmetic mean, we use LSE which is dominated by the largest distances.
    # This ensures the router focuses on the gate that is actually blocking progress.
    f_lse = 0.0
    n_f = len(self.front_layer)
    if n_f > 0:
        f_sum_exp = 0.0
        for g in self.front_layer:
            qubits = self.access2q[g]
            if not qubits: continue
            q1, q2 = qubits
            # Mapping state after the candidate swap is applied
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            
            # Square root of dependencies ensures critical paths have a strong but stable pull
            crit = math.sqrt(self.dag_dependencies_count[g] + 1.0)
            
            # Exp(dist) makes the cost grow exponentially with separation
            f_sum_exp += crit * math.exp(dist)
        
        # Taking the log brings the scale back to the distance domain
        f_lse = math.log(f_sum_exp) if f_sum_exp > 0 else 0.0
    
    # 3. Extended Layer: Sub-Linear Gravitational Pull
    # We use a cubic-root distance (dist^0.33) for the lookahead window.
    # This provides a broad "background" pull toward future partners that doesn't 
    # overwhelm the immediate bottleneck resolution.
    e_cost = 0.0
    n_e = len(self.extended_layer)
    if n_e > 0:
        for g in self.extended_layer:
            qubits = self.access2q[g]
            if not qubits: continue
            q1, q2 = qubits
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            
            crit = math.sqrt(self.dag_dependencies_count[g] + 1.0)
            depth = self.extended_layer_index.get(g, 0) + 1.0
            
            # Exponential decay for lookahead depth (sharper than harmonic, smoother than Gaussian)
            depth_weight = math.exp(-depth / 2.0)
            
            # Sub-linear distance penalty
            e_cost += (crit * (dist ** 0.333)) * depth_weight
        e_cost /= n_e

    # 4. Final Aggregation
    # We balance the sharp bottleneck focus with the broad lookahead signal.
    # The heat multiplier ensures we favor 'cool' hardware paths.
    W = 0.6  # Lookahead weight
    return heat_multiplier * (f_lse + W * e_cost)