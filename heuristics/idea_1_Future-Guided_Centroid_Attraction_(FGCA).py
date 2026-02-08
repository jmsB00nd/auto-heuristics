# Strategy: Future-Guided Centroid Attraction (FGCA)
# Intuition: Standard heuristics minimize current gate distances and lookahead distances independently, often leading to short-sighted moves. FGCA explicitly calculates a "future pull" for logical qubits in the front layer by measuring their distances to the specific physical locations of their future partners in the extended layer. This ensures that when a qubit moves to satisfy a current gate, it biased towards a direction that minimizes the distance to its next several interactions, reducing the total SWAP count.
# Stats: {'mean_swaps': 545.5454545454545, 'mean_depth': 925.1818181818181, 'mean_runtime': 4.638776887546886, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Physical qubits involved in the candidate SWAP
    p1, p2 = swap_gate[0], swap_gate[1]
    # max_decay captures the 'heat' or usage frequency of the physical qubits
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    f_cost = 0.0
    n_f = len(self.front_layer)
    if n_f == 0:
        return 0.0

    # 1. Front Layer Interaction + Future Pull
    # For each gate in the front layer, we calculate its current distance penalty
    # AND a "future pull" based on where its qubits need to be for upcoming gates.
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        
        dist = self.distance_matrix[Q1][Q2]
        crit = self.dag_dependencies_count[g] + 1
        
        # Immediate interaction cost (Quadratic to prioritize closer gates)
        f_cost += crit * (dist * dist)
        
        # Calculate "Future Pull" for q1 and q2
        # This looks ahead to see where these specific logical qubits are needed next.
        for ge in self.extended_layer:
            qe_pair = self.access2q[ge]
            if q1 in qe_pair:
                # Find the other logical qubit in the future gate
                q_other = qe_pair[1] if qe_pair[0] == q1 else qe_pair[0]
                Q_other = self.temp_mapping_dict[q_other]
                
                e_crit = self.dag_dependencies_count[ge] + 1
                e_depth = self.extended_layer_index.get(ge, 0) + 1
                
                # Weight future pull by criticality and decay it by depth
                f_cost += (e_crit * self.distance_matrix[Q1][Q_other]) / (e_depth * n_f)
                
            if q2 in qe_pair:
                q_other = qe_pair[1] if qe_pair[0] == q2 else qe_pair[0]
                Q_other = self.temp_mapping_dict[q_other]
                
                e_crit = self.dag_dependencies_count[ge] + 1
                e_depth = self.extended_layer_index.get(ge, 0) + 1
                
                f_cost += (e_crit * self.distance_matrix[Q2][Q_other]) / (e_depth * n_f)

    # 2. General Extended Layer Guidance (Global Pressure)
    # Provides a background signal for gates not directly connected to the front layer.
    e_cost = 0.0
    n_e = len(self.extended_layer)
    if n_e > 0:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            crit = self.dag_dependencies_count[g] + 1
            depth = self.extended_layer_index.get(g, 0) + 1
            e_cost += (crit * dist) / depth
        e_cost /= n_e

    # 3. Final Aggregation
    # Balance the specific "Future Pull" of front-layer qubits with the general "Global Pressure".
    W = 0.75 # Weight for global lookahead signal
    return max_decay * (f_cost / n_f + W * e_cost)