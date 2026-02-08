# Strategy: Saliency-Weighted Power-Law Potential (SWPP)
# Intuition: This heuristic treats the `dag_dependencies_count` as a "saliency" factor that weights the distance of a gate non-linearly (using a power of 1.5) to prioritize the most critical bottlenecks. For future gates, it applies a power-law decay based on lookahead depth to ensure that near-term interactions exert a much stronger "pull" than distant ones, while the `decay_parameter` scales the final cost to penalize repetitive use of the same physical qubits.
# Stats: {'mean_swaps': 597.9090909090909, 'mean_depth': 922.2272727272727, 'mean_runtime': 4.941670591180975, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

import math

def qlosure_poly_heuristic(self, swap_gate):
    # Retrieve the decay factor for the physical qubits involved in the swap
    # Using a slight non-linear scaling to penalize "hot" qubits more aggressively
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]]) ** 1.1

    # 1. Front Layer Scoring: Weighted by the square-root of criticality (saliency)
    # Higher criticality gates (larger transitive closure) dominate the cost.
    f_val = 0
    front_count = len(self.front_layer)
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        # temp_mapping_dict reflects the state AFTER the swap
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        # Saliency: (criticality + 1)^1.5 to prioritize the critical path
        saliency = (self.dag_dependencies_count[g] + 1) ** 1.5
        f_val += saliency * dist

    # Normalize front layer cost
    f_score = f_val / front_count if front_count > 0 else 0

    # 2. Extended Layer Scoring: Power-law depth decay
    # We use (depth + 1)^-1.2 to create a sharp distinction between immediate successors and distant lookahead.
    e_val = 0
    extended_count = len(self.extended_layer)
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        depth = self.extended_layer_index.get(g, 0)
        dist = self.distance_matrix[Q1][Q2]
        
        # Influence decays as a power-law of the lookahead depth
        depth_weight = 1.0 / ((depth + 1) ** 1.2)
        criticality = self.dag_dependencies_count[g] + 1
        
        e_val += criticality * dist * depth_weight

    # Normalize extended layer cost
    e_score = e_val / extended_count if extended_count > 0 else 0

    # Total Cost: Combined weighted distance scaled by physical qubit usage (decay)
    # A weight of 0.75 for the extended layer is used to balance current and future needs.
    W_ext = 0.75
    H = max_decay * (f_score + W_ext * e_score)

    return H