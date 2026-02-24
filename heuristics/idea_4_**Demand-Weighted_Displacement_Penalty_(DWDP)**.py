# Strategy: **Demand-Weighted Displacement Penalty (DWDP)**
# Intuition: Each logical qubit has a "demand centroid" — the weighted average of where it needs to be across all visible gates. Swaps that move a qubit closer to its demand centroid are good; swaps that move a high-demand qubit away from its centroid are catastrophic. By computing per-qubit demand pressure (how many gates need this qubit, weighted by criticality and urgency), we can penalize displacement of high-demand qubits while rewarding convergence toward demand centroids.
# Stats: {'mean_swaps': 689.1818181818181, 'mean_depth': 1083.5, 'mean_runtime': 7.047689806331288, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # --- Hyperparameters ---
    CRIT_PWR = 0.65          # Sub-linear criticality weighting
    DEPTH_DECAY = 0.55       # Gaussian depth decay for extended layer
    GAUSS_2SIG2 = 7.0        # 2*sigma^2 for Gaussian decay
    LOOKAHEAD_W = 0.65       # Weight for extended layer contribution
    CENTROID_W = 0.4         # Weight for demand-centroid displacement term
    DIST_EXP = 0.85          # Distance exponent (slightly sub-linear)

    p1, p2 = swap_gate
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # ---------------------------------------------------------------
    # Phase 1: Build per-logical-qubit "demand vectors"
    # For each logical qubit involved in visible gates, accumulate
    # a weighted demand for each partner's physical location.
    # The demand centroid for qubit q = weighted avg distance to
    # all partner locations it must interact with.
    # ---------------------------------------------------------------
    # demand_targets[logical_q] = list of (target_phys, weight)
    demand_targets = {}

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        deps = self.dag_dependencies_count[g]
        w = (deps + 1.0) ** CRIT_PWR

        # q1 needs to be near wherever q2 is mapped (and vice versa)
        Q2_after = self.temp_mapping_dict[q2]
        Q1_after = self.temp_mapping_dict[q1]

        if q1 not in demand_targets:
            demand_targets[q1] = []
        demand_targets[q1].append((Q2_after, w))

        if q2 not in demand_targets:
            demand_targets[q2] = []
        demand_targets[q2].append((Q1_after, w))

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        deps = self.dag_dependencies_count[g]
        depth = self.extended_layer_index.get(g, 0)
        decay = math.exp(-(depth ** 2) / GAUSS_2SIG2)
        w = (deps + 1.0) ** CRIT_PWR * decay * LOOKAHEAD_W

        Q2_after = self.temp_mapping_dict[q2]
        Q1_after = self.temp_mapping_dict[q1]

        if q1 not in demand_targets:
            demand_targets[q1] = []
        demand_targets[q1].append((Q2_after, w))

        if q2 not in demand_targets:
            demand_targets[q2] = []
        demand_targets[q2].append((Q1_after, w))

    # ---------------------------------------------------------------
    # Phase 2: Compute demand-centroid displacement for swapped qubits
    # For each logical qubit mapped to p1 or p2, measure how well
    # its current physical position serves its demand targets.
    # "Displacement" = weighted sum of distances from qubit's position
    # to all its demand targets.
    # ---------------------------------------------------------------
    centroid_cost = 0.0
    for log_q, targets in demand_targets.items():
        phys_q = self.temp_mapping_dict[log_q]
        total_w = 0.0
        weighted_dist = 0.0
        for target_phys, w in targets:
            d = self.distance_matrix[phys_q][target_phys]
            weighted_dist += w * (d ** DIST_EXP)
            total_w += w
        if total_w > 0:
            centroid_cost += weighted_dist

    # ---------------------------------------------------------------
    # Phase 3: Standard distance-based cost for front + extended layers
    # (complements the centroid term with direct gate distance scoring)
    # ---------------------------------------------------------------
    f_cost = 0.0
    f_count = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        deps = self.dag_dependencies_count[g]
        crit_w = (deps + 1.0) ** CRIT_PWR
        f_cost += crit_w * (dist ** DIST_EXP)
        f_count += 1

    e_cost = 0.0
    e_count = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        deps = self.dag_dependencies_count[g]
        depth = self.extended_layer_index.get(g, 0)
        decay = math.exp(-(depth ** 2) / GAUSS_2SIG2)
        crit_w = (deps + 1.0) ** CRIT_PWR
        e_cost += crit_w * (dist ** DIST_EXP) * decay
        e_count += 1

    # ---------------------------------------------------------------
    # Phase 4: Aggregation
    # ---------------------------------------------------------------
    h_f = (f_cost / f_count) if f_count > 0 else 0.0
    h_e = (e_cost / e_count) if e_count > 0 else 0.0

    gate_term = h_f + LOOKAHEAD_W * h_e

    # Normalize centroid cost by number of demand qubits
    num_demand = len(demand_targets) if demand_targets else 1
    centroid_term = centroid_cost / num_demand

    H = max_decay * (gate_term + CENTROID_W * centroid_term)
    return float(H)