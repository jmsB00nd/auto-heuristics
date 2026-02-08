# Strategy: "Geometric Mean Distance with Dependency Cascade Weighting"
# Intuition: Instead of arithmetic averaging, use geometric mean of distances which naturally penalizes outliers (gates far apart) more heavily. Combine this with a cascade factor that exponentially amplifies the importance of gates whose dependencies form long chains, creating urgency to resolve bottlenecks early.
# Stats: {'mean_swaps': 493.1363636363636, 'mean_depth': 1003.7272727272727, 'mean_runtime': 1.1338241533799605, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math
    
    # Compute cascade depth for each gate - how deep is the dependency chain below it
    cascade_weight = {}
    for g in self.front_layer:
        deps = self.dag_dependencies_count[g]
        # Cascade factor grows superlinearly with dependency count
        cascade_weight[g] = 1.0 + math.log1p(deps) ** 2
    
    for g in self.extended_layer:
        deps = self.dag_dependencies_count[g]
        layer_idx = self.extended_layer_index.get(g, 1) + 1
        # Dampen cascade importance for deeper lookahead gates
        cascade_weight[g] = (1.0 + math.log1p(deps) ** 2) / layer_idx
    
    # Geometric mean of front layer distances (weighted)
    front_log_sum = 0.0
    front_weight_sum = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        P1 = self.temp_mapping_dict[q1]
        P2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[P1][P2]
        w = cascade_weight[g]
        # Add 1 to distance to avoid log(0), then weight
        front_log_sum += w * math.log1p(dist)
        front_weight_sum += w
    
    front_geom = math.expm1(front_log_sum / front_weight_sum) if front_weight_sum > 0 else 0
    
    # Extended layer contribution with geometric aggregation
    ext_log_sum = 0.0
    ext_weight_sum = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        P1 = self.temp_mapping_dict[q1]
        P2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[P1][P2]
        w = cascade_weight[g]
        ext_log_sum += w * math.log1p(dist)
        ext_weight_sum += w
    
    ext_geom = math.expm1(ext_log_sum / ext_weight_sum) if ext_weight_sum > 0 else 0
    
    # Qubit locality penalty - favor swaps that don't disturb "cold" qubits
    p1, p2 = swap_gate
    swap_heat = self.decay_parameter[p1] + self.decay_parameter[p2]
    locality_factor = 1.0 + 0.1 * swap_heat
    
    # Combine with emphasis on front layer
    H = locality_factor * (front_geom + 0.5 * ext_geom)
    
    return H