# Strategy: **Coulombic Potential Energy Minimization**
# Intuition: Model each pending 2-qubit gate as an electrostatic bond whose potential energy is proportional to `criticality * dist^2 / (dist + 1)` — a saturated quadratic that aggressively penalizes nearby-but-not-adjacent pairs (the "last mile" problem) while bounding far-away contributions. The inverse-square *force* naturally emerges as the gradient of this potential, focusing routing effort where marginal improvement is greatest.
# Stats: {'mean_swaps': 758.5, 'mean_depth': 1023.9090909090909, 'mean_runtime': 2.9043517979708584, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front Layer: Coulombic Potential with Saturated Quadratic ---
    # Potential V(d) = d^2 / (d + 1) gives:
    #   d=0 -> 0, d=1 -> 0.5, d=2 -> 1.33, d=3 -> 2.25, d=5 -> 4.17
    # The derivative dV/dd = d(d+2)/(d+1)^2 peaks near d=1-2,
    # creating strong gradient exactly where routing progress matters most.
    front_potential = 0.0
    max_front_potential = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]

        deps = self.dag_dependencies_count[g]
        # Sub-linear criticality to prevent outlier dominance
        charge = (deps + 1) ** 0.5

        # Saturated quadratic potential
        potential = charge * (dist * dist) / (dist + 1.0) if dist > 0 else 0.0
        front_potential += potential

        # Track worst bottleneck for minimax component
        if potential > max_front_potential:
            max_front_potential = potential

    # Blend average + bottleneck: prevents ignoring critical gates
    avg_front = front_potential / front_layer_size
    front_score = 0.6 * avg_front + 0.4 * max_front_potential

    # --- Extended Layer: Harmonic-Decayed Coulombic Potential ---
    extended_potential = 0.0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]

        deps = self.dag_dependencies_count[g]
        charge = (deps + 1) ** 0.4  # Even gentler for lookahead

        depth = self.extended_layer_index.get(g, 0)
        # Exponential decay: 0.65^depth sharply discounts distant futures
        temporal_decay = 0.65 ** depth

        potential = charge * (dist * dist) / (dist + 1.0) * temporal_decay if dist > 0 else 0.0
        extended_potential += potential

    avg_extended = (extended_potential / extended_layer_size) if extended_layer_size > 0 else 0.0

    # --- Combine with calibrated weights ---
    W_ext = 0.5  # Extended layer contributes but doesn't dominate
    H = max_decay * (front_score + W_ext * avg_extended)

    return H