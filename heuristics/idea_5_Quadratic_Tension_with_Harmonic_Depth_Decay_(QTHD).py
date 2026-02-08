# Strategy: Quadratic Tension with Harmonic Depth Decay (QTHD)
# Intuition: This function utilizes a 1.5-power-law scaling on gate dependencies to amplify the "elastic tension" of the critical path, effectively forcing the router to prioritize major bottlenecks. It complements this with a geometric decay for the lookahead layer, which sharper than linear decay, to filter out topological "noise" from the distant future.
# Stats: {'mean_swaps': 581.4090909090909, 'mean_depth': 945.7727272727273, 'mean_runtime': 3.261225472797047, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Parameters for the heuristic
    POWER = 1.5        # Super-linear scaling for criticality
    LOOKAHEAD_W = 1.0  # Weight for the extended layer
    DECAY_RATE = 0.6   # Geometric decay for lookahead depth

    f_size = len(self.front_layer)
    e_size = len(self.extended_layer)

    # Calculate tension for the front layer (immediate gates)
    f_tension = 0
    for gate_id in self.front_layer:
        # access2q returns logical qubits, we map them to current physical locations
        q_logicals = self.access2q[gate_id]
        if not q_logicals: continue
        q_phys = [self.temp_mapping_dict[ql] for ql in q_logicals]
        
        # Power-law criticality weighting
        criticality = self.dag_dependencies_count[gate_id]
        dist = self.distance_matrix[q_phys[0]][q_phys[1]]
        f_tension += ((criticality + 1) ** POWER) * dist

    # Calculate tension for the extended layer (lookahead window)
    e_tension = 0
    for gate_id in self.extended_layer:
        q_logicals = self.access2q[gate_id]
        if not q_logicals: continue
        q_phys = [self.temp_mapping_dict[ql] for ql in q_logicals]
        
        # Geometric decay based on lookahead depth
        depth = self.extended_layer_index.get(gate_id, 0)
        criticality = self.dag_dependencies_count[gate_id]
        dist = self.distance_matrix[q_phys[0]][q_phys[1]]
        e_tension += (criticality + 1) * dist * (DECAY_RATE ** depth)

    # Normalize by layer sizes to maintain score stability
    h_front = f_tension / f_size if f_size > 0 else 0
    h_ext = e_tension / e_size if e_size > 0 else 0

    # Apply heat/decay penalty for the physical qubits involved in the candidate SWAP
    # This prevents the router from thrashing the same qubits repeatedly.
    swap_penalty = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Final combined cost (minimization)
    return swap_penalty * (h_front + LOOKAHEAD_W * h_ext)