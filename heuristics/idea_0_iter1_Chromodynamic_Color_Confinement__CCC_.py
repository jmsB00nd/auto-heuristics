# Idea: Chromodynamic Color Confinement (CCC)
# Stats: {"mean_swaps": 717.3181818181819, "mean_depth": 1100.8636363636363, "mean_runtime": 1.9365198070352727, "total_circuits": 22, "successful_runs": 22, "failed_runs": 0, "error": null, "first_failure_error": null, "first_failure_traceback": null}

def qlosure_poly_heuristic(self, swap_gate):
    # Idea: Chromodynamic Color Confinement (CCC)
    # Description: Models interaction cost as a "confining potential" (linear at long distances,
    # asymptotically free at short distances) weighted by "color charges" derived from 
    # sub-circuit disjointness (exclusivity).

    # --- Constants ---
    LOOKAHEAD_LIMIT = 40       # Window size for determining color charges
    ALPHA = 1.0                # Softening parameter: V(r) ~ sqrt(r^2 + ALPHA)
    COLOR_STRENGTH = 3.0       # Gain factor for "color singlet" (exclusive) pairs
    DECAY_FACTOR = 0.85        # Attenuation of force over circuit depth
    FRONT_WEIGHT = 10.0        # High weight for immediate "strong force" interactions

    # --- 1. Color Charge Profiling ---
    # "Color Charge" is assigned based on the exclusivity of qubit interactions in the lookahead.
    # Qubits forming disjoint sub-circuits (high exclusivity) are treated as "color singlets"
    # and experience a stronger confining force to prevent separation ("quark confinement").

    qubit_activity = {}
    pair_activity = {}

    # Analyze a limited window for efficiency
    window = self.extended_layer[:LOOKAHEAD_LIMIT]

    for g in window:
        q1, q2 = self.access2q[g]
        qubit_activity[q1] = qubit_activity.get(q1, 0) + 1
        qubit_activity[q2] = qubit_activity.get(q2, 0) + 1

        # Canonical pair key
        p_key = (q1, q2) if q1 < q2 else (q2, q1)
        pair_activity[p_key] = pair_activity.get(p_key, 0) + 1

    total_cost = 0.0

    # --- 2. Front Layer (Immediate Interaction) ---
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]

        # Potential V(r) = sqrt(r^2 + ALPHA)
        # Properties:
        # - Asymptotically Free: Slope -> 0 as r -> 0 (Weak force at short range)
        # - Confining: Slope -> 1 as r -> inf (Linear potential at long range)
        potential = (dist**2 + ALPHA) ** 0.5
        total_cost += FRONT_WEIGHT * potential

    # --- 3. Extended Layer (Confining Potential) ---
    for g in window:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]

        # Skip if not mapped (safety)
        if Q1 == -1 or Q2 == -1:
            continue

        dist = self.distance_matrix[Q1][Q2]

        # Calculate Exclusivity (Jaccard-like index)
        # Intersection / Union of lookahead participation
        n1 = qubit_activity.get(q1, 1)
        n2 = qubit_activity.get(q2, 1)
        p_key = (q1, q2) if q1 < q2 else (q2, q1)
        n12 = pair_activity.get(p_key, 0)

        # exclusivity -> 1.0 implies q1 and q2 only interact with each other (Disjoint sub-circuit)
        exclusivity = (2.0 * n12) / (n1 + n2 + 0.01)

        # "Color Charge" magnitude scales with exclusivity
        charge = 1.0 + COLOR_STRENGTH * exclusivity

        # Temporal decay
        depth = self.extended_layer_index.get(g, 0)
        decay = DECAY_FACTOR ** depth

        # Apply Confining Potential
        potential = (dist**2 + ALPHA) ** 0.5

        total_cost += charge * potential * decay

    return total_cost