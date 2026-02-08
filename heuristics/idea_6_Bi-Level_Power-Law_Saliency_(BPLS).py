# Strategy: Bi-Level Power-Law Saliency (BPLS)
# Intuition: This cost function prioritizes immediate bottlenecks by applying a quadratic weight to the criticality of front-layer gates, while using a sub-linear (square root) distance scaling for the lookahead layer to prevent the router from over-committing to distant logical pairs that are not yet ready for execution.
# Stats: {'mean_swaps': 583.2727272727273, 'mean_depth': 931.9090909090909, 'mean_runtime': 2.624314719980413, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    front_size = len(self.front_layer)
    extended_size = len(self.extended_layer)
    if front_size == 0:
        return 0

    # Physical qubit decay penalty (prevents thrashing/overuse of specific qubits)
    q1_phys, q2_phys = swap_gate
    max_decay = max(self.decay_parameter[q1_phys], self.decay_parameter[q2_phys])

    # Front Layer: Quadratic criticality to force immediate resolution of major bottlenecks
    f_tension = 0
    for g_id in self.front_layer:
        l_qubits = self.access2q[g_id]
        if not l_qubits:
            continue
        lq1, lq2 = l_qubits
        
        pq1, pq2 = self.temp_mapping_dict[lq1], self.temp_mapping_dict[lq2]
        dist = self.distance_matrix[pq1][pq2]
        
        # Saliency is the square of dependencies to ensure bottleneck gates dominate
        saliency = (self.dag_dependencies_count[g_id] + 1) ** 2
        f_tension += saliency * dist

    # Extended Layer: Sub-linear distance and quadratic depth decay
    e_tension = 0
    if extended_size > 0:
        for g_id in self.extended_layer:
            l_qubits = self.access2q[g_id]
            if not l_qubits:
                continue
            lq1, lq2 = l_qubits
            
            pq1, pq2 = self.temp_mapping_dict[lq1], self.temp_mapping_dict[lq2]
            dist = self.distance_matrix[pq1][pq2]
            
            # Lookahead depth factor (1, 2, 3...)
            depth = self.extended_layer_index.get(g_id, 0) + 1
            criticality = self.dag_dependencies_count[g_id]
            
            # Use square root of distance to focus on "near-ready" lookahead gates
            # and quadratic depth decay to limit the influence of far-future gates
            e_tension += (criticality + 1) * (dist ** 0.5) / (depth ** 2)

    # Combine using the front layer as the primary driver, weighted by decay
    cost = max_decay * (f_tension / front_size + (e_tension / extended_size if extended_size else 0))
    
    return cost