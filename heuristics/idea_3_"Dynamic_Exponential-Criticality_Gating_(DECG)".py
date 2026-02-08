# Strategy: "Dynamic Exponential-Criticality Gating (DECG)"
# Intuition: This heuristic uses an exponential gating function based on the relative criticality of gates compared to the current average bottleneck size. By weighting distances with a $2^{(\text{crit}/\text{avg\_crit})}$ factor, it dynamically identifies and prioritizes the most restrictive gates in the dependency graph, while applying a quadratic temporal decay to future operations to prevent "lookahead noise" from overriding immediate routing needs.
# Stats: {'mean_swaps': 762.5454545454545, 'mean_depth': 1001.6818181818181, 'mean_runtime': 3.365059321576899, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

import math

def qlosure_poly_heuristic(self, swap_gate):
    # 1. Contextual Feature Engineering: Derived Average Criticality
    # We calculate the mean dependency count of the current window to establish a 'bottleneck baseline'
    all_active_gates = list(self.front_layer) + self.extended_layer
    if not all_active_gates:
        return 0.0
    
    avg_crit = sum(self.dag_dependencies_count[g] for g in all_active_gates) / len(all_active_gates)
    if avg_crit == 0:
        avg_crit = 1.0

    # 2. Physical Qubit Health (Heat Factor)
    # Penalize swaps on physical qubits with high decay/noise
    heat_factor = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # 3. Front Layer Cost: Exponentially-Gated Tension
    # We use a power-law distance (1.5) to penalize long-range separations more than linearly
    f_total_cost = 0.0
    for g in self.front_layer:
        q1_log, q2_log = self.access2q[g]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        dist = self.distance_matrix[p1][p2]
        
        # Exponential gating: Gates significantly above avg_crit gain massive priority
        # This forces the router to resolve the "true" bottlenecks first.
        relative_crit = self.dag_dependencies_count[g] / avg_crit
        weight = 2.0 ** relative_crit
        f_total_cost += weight * (dist ** 1.5)

    # 4. Extended Layer Cost: Quadratic Temporal Decay
    # Future gates are weighted by their distance and criticality, 
    # but their influence vanishes quadratically with their depth in the lookahead window.
    e_total_cost = 0.0
    for g in self.extended_layer:
        q1_log, q2_log = self.access2q[g]
        p1 = self.temp_mapping_dict[q1_log]
        p2 = self.temp_mapping_dict[q2_log]
        dist = self.distance_matrix[p1][p2]
        
        depth = self.extended_layer_index.get(g, 0) + 1
        relative_crit = self.dag_dependencies_count[g] / avg_crit
        weight = 2.0 ** relative_crit
        
        # Quadratic decay ensures immediate successors matter much more than distant ones
        e_total_cost += (weight * dist) / (depth ** 2)

    # 5. Final Score Aggregation
    # Normalize by layer sizes to maintain stability across different circuit widths
    f_size = len(self.front_layer) if self.front_layer else 1
    e_size = len(self.extended_layer) if self.extended_layer else 1
    
    # We apply a 0.4 lookahead constant (W) to balance current vs future tension
    H = heat_factor * ((f_total_cost / f_size) + 0.4 * (e_total_cost / e_size))

    return float(H)