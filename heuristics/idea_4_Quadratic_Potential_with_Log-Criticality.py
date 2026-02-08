# Strategy: Quadratic Potential with Log-Criticality
# Intuition: This heuristic models qubit connections as physical springs with quadratic potential energy (`distance^2`), which penalizes large separations (outliers) significantly more than the standard linear model, forcing the router to eliminate "stragglers". It concurrently dampens the dependency count using a logarithmic scale (`log(deps)`) to prevent high-criticality gates from completely dominating the cost function, ensuring a more balanced optimization between immediate locality and future structure.
# Stats: {'mean_swaps': 578.6818181818181, 'mean_depth': 969.7727272727273, 'mean_runtime': 1.4502952857451006, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # Hyperparameters
    W_lookahead = 0.5   # Weighting for the extended layer relative to front
    DECAY_BASE = 0.85   # Exponential decay base for lookahead depth

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # --- Front Layer: Quadratic Potential ---
    # We use distance^2 to heavily penalize the "worst-case" links (stragglers).
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        # Logarithmic criticality to dampen the effect of massive dependency chains
        deps = self.dag_dependencies_count[g]
        crit_weight = math.log(deps + 2.0)
        
        dist = self.distance_matrix[Q1][Q2]
        f_cost += crit_weight * (dist ** 2)

    # Normalize front layer cost
    if front_layer_size > 0:
        f_cost /= front_layer_size

    # --- Extended Layer: Exponentially Decayed Linear Potential ---
    # For future gates, we use linear distance (less aggressive) and exponential decay.
    e_cost = 0.0
    if extended_layer_size > 0:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            
            # Exponential decay based on lookahead depth
            layer_depth = self.extended_layer_index.get(g, 0)
            depth_weight = DECAY_BASE ** layer_depth
            
            deps = self.dag_dependencies_count[g]
            crit_weight = math.log(deps + 2.0)
            
            dist = self.distance_matrix[Q1][Q2]
            e_cost += crit_weight * dist * depth_weight
            
        e_cost /= extended_layer_size

    # Total Heuristic Score
    H = max_decay * (f_cost + W_lookahead * e_cost)

    return H