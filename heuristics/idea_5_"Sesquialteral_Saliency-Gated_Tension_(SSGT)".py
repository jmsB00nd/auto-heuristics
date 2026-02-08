# Strategy: "Sesquialteral Saliency-Gated Tension (SSGT)"
# Intuition: This function employs a "sesquialteral" (3/2) power law for physical distances to create a cost landscape that is more punitive than linear distance but more stable than quadratic growth, effectively balancing greedy local moves with long-term routing. It weights each gate by the square root of its transitive dependency count ("Saliency") to prioritize the critical path while preventing high-dependency bottlenecks from completely overwhelming the lookahead signals.
# Stats: {'mean_swaps': 760.4090909090909, 'mean_depth': 1027.4545454545455, 'mean_runtime': 2.0485494353554468, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    # Parameter for lookahead weighting (Decaying Future Horizon)
    W = 0.8
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # viscosity represents the "heat" or noise penalty on the physical qubits involved
    viscosity = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # 1. Evaluate Front Layer (Immediate Targets)
    f_tension = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        phys_q1, phys_q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[phys_q1][phys_q2]
        
        # Saliency: Diminishing returns on dependency count to prevent outlier dominance
        saliency = (self.dag_dependencies_count[g] + 1) ** 0.5
        
        # Sesquialteral Tension: d^1.5 provides a super-linear penalty that smooths convergence
        f_tension += saliency * (dist ** 1.5)

    # 2. Evaluate Extended Layer (Lookahead Horizon)
    e_tension = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        phys_q1, phys_q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[phys_q1][phys_q2]
        
        # depth: 1 for immediate successors, higher for distant ones
        depth_factor = self.extended_layer_index.get(g, 0) + 1
        saliency = (self.dag_dependencies_count[g] + 1) ** 0.5
        
        # Apply harmonic depth decay to the lookahead tension
        e_tension += (saliency * (dist ** 1.5)) / depth_factor

    # 3. Normalize and Aggregate
    # Calculate average scores to ensure the heuristic scales with circuit width
    f_score = f_tension / front_layer_size if front_layer_size > 0 else 0
    e_score = e_tension / extended_layer_size if extended_layer_size > 0 else 0

    # The final cost H combines present tension and future potential, scaled by qubit health
    H = viscosity * (f_score + W * e_score)

    return H