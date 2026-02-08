# Strategy: Gaussian-Gated Lookahead (GGL)
# Intuition: Most heuristics use harmonic or exponential decay for the lookahead window, which either lets distant noise bleed in or suppresses the immediate future too quickly. Gaussian gating ($e^{-d^2/2\sigma^2}$) provides a high-fidelity "planning horizon" that treats the next few layers as nearly as important as the front layer before rapidly tapering off, while sub-linear criticality weighting ($deps^{0.75}$) ensures the router remains sensitive to parallel branches rather than hyper-fixating on a single path.
# Stats: {'mean_swaps': 499.95454545454544, 'mean_depth': 936.0454545454545, 'mean_runtime': 2.1101507490331475, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    # Parameters for Gaussian decay and lookahead weighting
    # sigma=2.0 provides a smooth decay over the first few lookahead levels
    sigma_sq_2 = 8.0  # Equivalent to 2 * (sigma^2)
    lookahead_weight = 0.75
    
    # Retrieve the physical qubits for the candidate swap to factor in usage "heat"
    p1, p2 = swap_gate[0], swap_gate[1]
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])
    
    # 1. Front Layer: Sub-Linear Criticality Weighting
    # We use a 0.75 power to balance the urgency of deep dependency chains 
    # against the need to maintain progress on parallel circuit branches.
    f_score, f_count = 0.0, 0
    for g in self.front_layer:
        qubits = self.access2q[g]
        if not qubits: continue
        q1, q2 = qubits
        # Get physical locations after applying the candidate SWAP
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        crit = (self.dag_dependencies_count[g] + 1.0) ** 0.75
        f_score += crit * self.distance_matrix[Q1][Q2]
        f_count += 1
        
    # 2. Extended Layer: Gaussian-Gated Decay
    # Future interactions are filtered through a Gaussian kernel to maintain 
    # a strong "attentional focus" on the immediate successors.
    e_score, e_count = 0.0, 0
    for g in self.extended_layer:
        qubits = self.access2q[g]
        if not qubits: continue
        q1, q2 = qubits
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        depth = self.extended_layer_index.get(g, 0)
        crit = (self.dag_dependencies_count[g] + 1.0) ** 0.75
        
        # Gaussian gating: e^(-depth^2 / 2sigma^2)
        decay = math.exp(-(depth**2) / sigma_sq_2)
        e_score += crit * self.distance_matrix[Q1][Q2] * decay
        e_count += 1
        
    # Normalization to ensure stability regardless of layer density
    h_f = (f_score / f_count) if f_count > 0 else 0.0
    h_e = (e_score / e_count) if e_count > 0 else 0.0
    
    # Combined cost: lower is better
    return max_decay * (h_f + lookahead_weight * h_e)