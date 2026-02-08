# Strategy: "Saturated Criticality with Global Gravity"
# Intuition: Linear cost functions treat all distance reductions equally (e.g., 10→9 vs 1→0), but only reaching 0 enables execution. This heuristic uses a saturated cost function `dist / (dist + 1)` for the front layer to create a steep gradient near zero (strongly incentivizing immediate execution) while flattening out for distant pairs. This allows the "Global Gravity" of the extended layer (measured linearly) to guide the long-range structure, preventing the router from over-optimizing non-urgent distant gates in the front layer.
# Stats: {'mean_swaps': 492.3636363636364, 'mean_depth': 1025.0454545454545, 'mean_runtime': 1.1700791987505825, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Front Layer: Saturated Cost Optimization
    # Metric: dist / (dist + 1)
    # Creates a steep gradient for near-term gates (prioritizing 1->0 execution)
    # while capping the penalty for far-away gates (preventing distraction).
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        crit = self.dag_dependencies_count[g] + 1
        
        # Saturated distance term [0, 1)
        term = dist / (dist + 1.0)
        f_cost += crit * term

    front_layer_size = len(self.front_layer)
    f_norm = f_cost / front_layer_size if front_layer_size else 0

    # Extended Layer: Linear Cost (Global Gravity)
    # Metric: dist
    # Maintains linear "pull" for future gates to preserve global structure.
    e_cost = 0.0
    extended_layer_size = len(self.extended_layer)
    if extended_layer_size:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            crit = self.dag_dependencies_count[g] + 1
            depth = self.extended_layer_index.get(g, 0) + 1
            
            # Linear distance weighted by inverse-square depth
            e_cost += (crit * dist) / (depth ** 2)
        
        e_norm = e_cost / extended_layer_size
    else:
        e_norm = 0

    W = 0.5
    H = max_decay * (f_norm + W * e_norm)

    return H