# Strategy: Elastic Bottleneck Tension (EBT)
# Intuition: This heuristic treats qubit pairs as if they are connected by non-linear elastic springs where the "stiffness" is determined by the gate's criticality (transitive closure size). By squaring the physical distance, we apply a much harsher penalty to qubits that drift apart, while the power-law scaling of dependencies ensures that the router prioritize resolving "high-pressure" bottlenecks that block the largest portions of the DAG.
# Stats: {'mean_swaps': 528.4090909090909, 'mean_depth': 928.4090909090909, 'mean_runtime': 3.6129779382185503, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # 1. Access the decay parameter to prevent physical qubit overuse (heat management)
    # swap_gate is (phys_q0, phys_q1)
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # 2. Front Layer Tension: Immediate interaction pressure
    f_sum = 0
    f_count = 0
    for g in self.front_layer:
        l_qubits = self.access2q[g]
        if len(l_qubits) < 2:
            continue
        
        # Mapping state after the candidate SWAP is already in temp_mapping_dict
        p1 = self.temp_mapping_dict[l_qubits[0]]
        p2 = self.temp_mapping_dict[l_qubits[1]]
        
        # dist is the shortest path between physical nodes
        dist = self.distance_matrix[p1][p2]
        
        # Criticality: Super-linear weighting of the transitive closure
        # This amplifies the importance of gates that unblock large sub-graphs
        criticality_weight = (self.dag_dependencies_count[g] + 1) ** 1.5
        
        # Elastic Tension: Stiffness * distance squared
        f_sum += criticality_weight * (dist ** 2)
        f_count += 1

    # 3. Extended Layer Tension: Future interaction guidance
    e_sum = 0
    e_count = 0
    for g in self.extended_layer:
        l_qubits = self.access2q[g]
        if len(l_qubits) < 2:
            continue
        
        p1 = self.temp_mapping_dict[l_qubits[0]]
        p2 = self.temp_mapping_dict[l_qubits[1]]
        dist = self.distance_matrix[p1][p2]
        
        # Lookahead depth: Geometric decay to prioritize near-future gates
        depth = self.extended_layer_index.get(g, 0)
        temporal_decay = 2.0 ** (depth + 1)
        
        criticality_weight = (self.dag_dependencies_count[g] + 1) ** 1.5
        e_sum += (criticality_weight * (dist ** 2)) / temporal_decay
        e_count += 1

    # 4. Normalization and Aggregation
    # We normalize by the count of 2-qubit gates in each layer to maintain scale
    f_score = f_sum / f_count if f_count > 0 else 0
    e_score = e_sum / e_count if e_count > 0 else 0

    # Total cost (Lower is better)
    # We use a weight of 1.0 for lookahead as the temporal_decay already handles scaling
    H = max_decay * (f_score + e_score)

    return H