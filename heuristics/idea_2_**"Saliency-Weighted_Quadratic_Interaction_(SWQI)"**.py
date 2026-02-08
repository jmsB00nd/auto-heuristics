# Strategy: **"Saliency-Weighted Quadratic Interaction (SWQI)"**
# Intuition: This heuristic treats the "tension" between qubits as an interaction energy that scales quadratically with physical distance to aggressively prioritize resolving long-range gates in the front layer. It utilizes a logarithmic transformation of the dependency count (via bit-length) to prevent deep critical paths from inducing "tunnel vision" in the router, while employing a power-law decay for the lookahead window to ensure future gates provide guidance without over-constraining the current local optimal move.
# Stats: {'mean_swaps': 727.7272727272727, 'mean_depth': 1038.8181818181818, 'mean_runtime': 1.6778755187988281, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Calculate physical qubit "viscosity" (decay/heat impact)
    # Moving through "hot" qubits is penalized to prevent localized congestion
    heat_penalty = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    f_total = 0
    front_size = len(self.front_layer)
    for g in self.front_layer:
        q1_log, q2_log = self.access2q[g]
        p1, p2 = self.temp_mapping_dict[q1_log], self.temp_mapping_dict[q2_log]
        dist = self.distance_matrix[p1][p2]
        
        # Saliency: Log-weighted criticality (using bit_length as a robust log2 proxy)
        # This prevents extremely large dependency counts from drowning out the logic
        saliency = (self.dag_dependencies_count[g] + 1).bit_length()
        
        # Quadratic distance penalty: Prioritize closing large gaps immediately
        f_total += saliency * (dist ** 2)

    e_total = 0
    ext_size = len(self.extended_layer)
    for g in self.extended_layer:
        q1_log, q2_log = self.access2q[g]
        p1, p2 = self.temp_mapping_dict[q1_log], self.temp_mapping_dict[q2_log]
        dist = self.distance_matrix[p1][p2]
        
        depth = self.extended_layer_index.get(g, 0)
        saliency = (self.dag_dependencies_count[g] + 1).bit_length()
        
        # Power-law decay for lookahead (1.5^-depth) 
        # Provides a smoother gradient than linear decay but less aggressive than 2^-n
        lookahead_weight = 1.0 / (1.5 ** (depth + 1))
        e_total += (saliency * dist) * lookahead_weight

    # Normalization and Fusion
    # We maintain the front layer as the primary driver, 
    # using the extended layer as a normalized tie-breaker.
    h_front = f_total / front_size if front_size > 0 else 0
    h_ext = e_total / ext_size if ext_size > 0 else 0
    
    # H = Viscosity * (Front_Tension + Extended_Guidance)
    cost = heat_penalty * (h_front + h_ext)
    
    return float(cost)