# Strategy: Anisotropic Cubic Minkowski Saliency (ACMS)
# Intuition: This heuristic utilizes a Cubic Minkowski norm ($L_3$ power mean) to aggregate routing tension, which disproportionately penalizes logical qubit pairs that remain far apart despite their high criticality. By transitioning from a linear average to a cubic aggregation, the router focuses aggressively on resolving the single most severe bottlenecks in the front layer rather than being distracted by small, easily resolvable improvements elsewhere.
# Stats: {'mean_swaps': 759.6363636363636, 'mean_depth': 1027.090909090909, 'mean_runtime': 2.6072115031155674, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # 1. Hyperparameters
    LOOKAHEAD_WEIGHT = 1.0
    MINKOWSKI_P = 3.0  # Cubic aggregation to focus on outlier bottlenecks
    
    # 2. Hardware Heat/Decay Factor
    # We factor in the "heat" of the physical qubits being swapped to avoid high-noise areas.
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    
    # 3. Front Layer Cost Calculation
    # We use the Log-Saliency of gates multiplied by their physical distance, 
    # then aggregate using a cubic mean (L3 norm).
    f_tension_p_sum = 0.0
    f_size = len(self.front_layer)
    
    for g_id in self.front_layer:
        q1_log, q2_log = self.access2q[g_id]
        p1, p2 = self.temp_mapping_dict[q1_log], self.temp_mapping_dict[q2_log]
        
        # Distance on hardware graph
        dist = self.distance_matrix[p1][p2]
        # Saliency: Logarithmic scaling of transitive closure to normalize DAG importance
        saliency = math.log(self.dag_dependencies_count[g_id] + 2)
        
        f_tension_p_sum += (saliency * dist) ** MINKOWSKI_P
        
    # Generalized Power Mean for Front Layer
    f_h = (f_tension_p_sum / f_size) ** (1.0 / MINKOWSKI_P) if f_size > 0 else 0.0
    
    # 4. Extended Layer Cost Calculation
    # Future gates are weighted by their depth in the lookahead window using exponential decay.
    e_tension_p_sum = 0.0
    e_size = len(self.extended_layer)
    
    for g_id in self.extended_layer:
        q1_log, q2_log = self.access2q[g_id]
        p1, p2 = self.temp_mapping_dict[q1_log], self.temp_mapping_dict[q2_log]
        
        dist = self.distance_matrix[p1][p2]
        
        # Depth decay: uses an exponential factor to decrease influence of distant future gates
        depth = self.extended_layer_index.get(g_id, 0)
        depth_decay = math.exp(-0.4 * depth)
        
        saliency = math.log(self.dag_dependencies_count[g_id] + 2)
        
        e_tension_p_sum += (depth_decay * saliency * dist) ** MINKOWSKI_P
        
    # Generalized Power Mean for Extended Layer
    e_h = (e_tension_p_sum / e_size) ** (1.0 / MINKOWSKI_P) if e_size > 0 else 0.0
    
    # 5. Result Composition
    # The final heuristic score is the noise-weighted sum of current and future routing tension.
    H = max_decay * (f_h + LOOKAHEAD_WEIGHT * e_h)
    
    return float(H)