# Strategy: Qubit-Centric Interaction Urgency (QCIU)
# Intuition: Circuit routing is often bottlenecked by "hub" qubits—logical qubits involved in multiple concurrent or near-term gates. While standard heuristics weight gates by criticality, they ignore the "Logical Degree" of the participating qubits in the interaction graph. QCIU calculates a saliency score for each logical qubit based on its involvement frequency across both the front and extended layers (decayed by depth). By weighting each gate's cost by the combined urgency of its participants, the router prioritizes the optimal placement of high-traffic qubits, effectively clearing multiple dependency paths simultaneously and reducing total SWAP overhead.
# Stats: {'mean_swaps': 600.2727272727273, 'mean_depth': 951.1818181818181, 'mean_runtime': 8.045862338759683, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Physical qubits involved in the candidate SWAP to penalize overuse (heat)
    p1, p2 = swap_gate[0], swap_gate[1]
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # 1. Calculate Logical Urgency (Interaction Degree) for each logical qubit.
    # We measure how "busy" each logical qubit is within the current lookahead window.
    # Hub qubits (those in many gates) are assigned higher urgency scores.
    logical_urgency = {} 
    
    # Process Front Layer (immediate interactions)
    for g in self.front_layer:
        for q in self.access2q[g]:
            logical_urgency[q] = logical_urgency.get(q, 0.0) + 1.0
            
    # Process Extended Layer (future interactions) with exponential depth decay
    for g in self.extended_layer:
        depth = self.extended_layer_index.get(g, 0)
        weight = 0.5 ** (depth + 1) # Half-life decay per lookahead step
        for q in self.access2q[g]:
            logical_urgency[q] = logical_urgency.get(q, 0.0) + weight

    # 2. Front Layer: Weighted Quadratic Tension
    f_cost = 0.0
    n_f = len(self.front_layer)
    if n_f > 0:
        for g in self.front_layer:
            qubits = self.access2q[g]
            if len(qubits) < 2: continue
            q1, q2 = qubits
            # Mapping state after the candidate swap is applied
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            crit = self.dag_dependencies_count[g] + 1.0
            
            # Hub weight: prioritize gates whose qubits are needed for other interactions
            hub_weight = logical_urgency.get(q1, 0.0) + logical_urgency.get(q2, 0.0)
            f_cost += (hub_weight * crit) * (dist ** 2)
        f_cost /= n_f

    # 3. Extended Layer: Weighted Linear Tension with Quadratic Depth Decay
    e_cost = 0.0
    n_e = len(self.extended_layer)
    if n_e > 0:
        for g in self.extended_layer:
            qubits = self.access2q[g]
            if len(qubits) < 2: continue
            q1, q2 = qubits
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            dist = self.distance_matrix[Q1][Q2]
            crit = self.dag_dependencies_count[g] + 1.0
            depth = self.extended_layer_index.get(g, 0) + 1.0
            
            hub_weight = logical_urgency.get(q1, 0.0) + logical_urgency.get(q2, 0.0)
            # Use linear distance for lookahead to avoid over-reacting to distant gates
            e_cost += (hub_weight * crit * dist) / (depth ** 2)
        e_cost /= n_e

    # 4. Aggregation
    # Balance immediate execution needs with lookahead guidance (W=0.5)
    # Scaled by the physical qubit usage heat to prevent congestion.
    return max_decay * (f_cost + 0.5 * e_cost)