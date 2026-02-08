# Strategy: "Log-Transitive Sesquialteral Tension (LTST)"
# Intuition: This function utilizes a logarithmic scaling for transitive dependencies to prevent deep circuit branches from dominating the routing priority, combined with a sesquialteral (1.5-power) distance penalty to aggressively discourage qubit dispersion. It employs an exponential depth decay for the lookahead window, creating a "focal point" on the immediate frontier while maintaining a non-linear sensitivity to physical qubit "heat" (decay).
# Stats: {'mean_swaps': 736.5909090909091, 'mean_depth': 1024.4545454545455, 'mean_runtime': 2.2218877510590986, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # Constants for non-linear scaling
    E_REF = 2.718281828  # Natural base for log-saliency offset
    DIST_POW = 1.5       # Sesquialteral power for distance penalty
    LOOKAHEAD_BASE = 1.6 # Exponential base for lookahead attenuation
    HEAT_AMP = 1.25      # Amplification factor for physical qubit decay

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # Calculate physical qubit "heat" impact
    # decay_parameter represents noise/reliability of the physical nodes
    max_decay = math.pow(max(1.0, self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]]), HEAT_AMP)

    # 1. Front Layer Potential
    f_potential = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        # Saliency: Sub-linear growth via log of transitive closure
        # Prevents a single massive dependency chain from monopolizing the heuristic
        saliency = math.log(E_REF + self.dag_dependencies_count[g])
        
        # Distance: Super-linear penalty to prioritize closing large gaps
        dist = self.distance_matrix[Q1][Q2]
        f_potential += saliency * math.pow(dist, DIST_POW)

    f_score = f_potential / front_layer_size if front_layer_size else 0

    # 2. Extended Layer Potential (Lookahead)
    e_potential = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        # Depth-based exponential decay
        depth = self.extended_layer_index.get(g, 0)
        depth_factor = math.pow(LOOKAHEAD_BASE, depth + 1)
        
        saliency = math.log(E_REF + self.dag_dependencies_count[g])
        dist = self.distance_matrix[Q1][Q2]
        
        # Combine saliency, distance, and depth attenuation
        e_potential += (saliency * math.pow(dist, DIST_POW)) / depth_factor

    e_score = e_potential / extended_layer_size if extended_layer_size else 0

    # 3. Final Heuristic Aggregation
    # H = Heat * (Front_Pressure + Extended_Pressure)
    H = max_decay * (f_score + e_score)

    return float(H)