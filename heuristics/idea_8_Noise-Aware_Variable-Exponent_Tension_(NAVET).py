# Strategy: Noise-Aware Variable-Exponent Tension (NAVET)
# Intuition: This heuristic introduces a dynamic distance exponent that increases as logical qubits move further apart, creating a "Super-Quadratic" penalty for long-range bottlenecks. It simultaneously incorporates local qubit decay (noise) into each gate's cost, ensuring that critical dependency paths are routed through reliable hardware regions. By scaling the tension power based on the separation distance, the router remains flexible for nearby qubits but applies massive pressure to resolve distant interactions that threaten circuit depth.
# Stats: {'mean_swaps': 559.5454545454545, 'mean_depth': 918.0, 'mean_runtime': 1.9286534786224365, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # 1. Front Layer: Noise-Aware Variable Tension
    # We use a variable exponent to penalize long-range moves more aggressively
    # than short-range ones, while weighting by local physical noise.
    f_total = 0.0
    f_size = len(self.front_layer)
    if f_size > 0:
        for g in self.front_layer:
            qs = self.access2q[g]
            if len(qs) < 2: continue
            q1, q2 = qs
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            crit = self.dag_dependencies_count[g] + 1.0
            
            # Local noise factor: penalize mapping critical gates to "hot" physical qubits
            noise_penalty = 1.0 + 2.0 * (self.decay_parameter[Q1] + self.decay_parameter[Q2])
            
            # Variable exponent: starts at 1.6 (sub-quadratic) for short range,
            # but scales linearly with distance (e.g., p=2.6 at dist=10).
            # This creates a "hardening" effect for distant logical qubits.
            p = 1.6 + (dist / 10.0)
            f_total += crit * (dist ** p) * noise_penalty
        h_f = f_total / f_size
    else:
        h_f = 0.0

    # 2. Extended Layer: Quadratic Depth Decay
    # We use linear distance for the lookahead window to provide a smooth pull,
    # but apply a quadratic decay for depth to prioritize the immediate future.
    e_total = 0.0
    e_size = len(self.extended_layer)
    if e_size > 0:
        for g in self.extended_layer:
            qs = self.access2q[g]
            if len(qs) < 2: continue
            q1, q2 = qs
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            crit = self.dag_dependencies_count[g] + 1.0
            depth = self.extended_layer_index.get(g, 0) + 1.0
            
            # Harmonic/Quadratic decay ensures future gates guide rather than distract
            e_total += (crit * dist) / (depth ** 2)
        h_e = e_total / e_size
    else:
        h_e = 0.0

    # 3. Aggregation and Candidate Swap Heat Multiplier
    # The global decay parameter of the swap qubits prevents high-frequency "chatter"
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    # Combined score (minimization objective)
    return max_decay * (h_f + 0.4 * h_e)