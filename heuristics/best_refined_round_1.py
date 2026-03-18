def qlosure_poly_heuristic(self, swap_gate):
    LARGE_PENALTY    = 1e9
    SWAP_COST        = 3.0
    RESOLUTION_MULT  = 1.5   # Bonus when a gate becomes immediately executable (dist→1)
    LOOKAHEAD_W      = 0.4

    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # ------------------------------------------------------------------ #
    # 1. Front layer: quadratic criticality-weighted improvement           #
    #    Normalise by total weight, not gate count, so critical gates      #
    #    dominate; resolution bonus rewards swaps that unblock gates now.  #
    # ------------------------------------------------------------------ #
    front_improvement  = 0.0
    total_front_weight = 0.0

    for g in self.front_layer:
        q1, q2      = self.access2q[g]
        dist_before = self.distance_matrix[self.mapping_dict[q1]][self.mapping_dict[q2]]
        dist_after  = self.distance_matrix[self.temp_mapping_dict[q1]][self.temp_mapping_dict[q2]]
        criticality = (self.dag_dependencies_count[g] + 1) ** 2   # quadratic emphasis
        delta       = dist_before - dist_after

        # Gate becomes adjacent → executable on next round: amplify signal
        if dist_after == 1 and dist_before > 1:
            delta *= RESOLUTION_MULT

        front_improvement  += criticality * delta
        total_front_weight += criticality

    front_norm = front_improvement / total_front_weight if total_front_weight > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 2. Extended layer: depth-discounted, criticality-weighted, also     #
    #    normalised by total weight rather than raw gate count.           #
    # ------------------------------------------------------------------ #
    ext_improvement  = 0.0
    total_ext_weight = 0.0

    for g in self.extended_layer:
        q1, q2      = self.access2q[g]
        dist_before = self.distance_matrix[self.mapping_dict[q1]][self.mapping_dict[q2]]
        dist_after  = self.distance_matrix[self.temp_mapping_dict[q1]][self.temp_mapping_dict[q2]]
        depth       = self.extended_layer_index.get(g, 0) + 1
        w           = (self.dag_dependencies_count[g] + 1) / depth
        ext_improvement  += w * (dist_before - dist_after)
        total_ext_weight += w

    ext_norm = ext_improvement / total_ext_weight if total_ext_weight > 0 else 0.0

    # ------------------------------------------------------------------ #
    # 3. Ratio cost: lower is better; non-improving swaps hard-penalised. #
    # ------------------------------------------------------------------ #
    net_improvement = front_norm + LOOKAHEAD_W * ext_norm

    if net_improvement <= 0.0:
        cost = LARGE_PENALTY
    else:
        cost = SWAP_COST / net_improvement

    # ------------------------------------------------------------------ #
    # 4. Anti-oscillation decay (unchanged from SPRC).                    #
    # ------------------------------------------------------------------ #
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    cost *= max_decay

    # ------------------------------------------------------------------ #
    # 5. Deterministic lexicographic tie-breaker.                         #
    # ------------------------------------------------------------------ #
    cost += (swap_gate[0] * self.num_qubits + swap_gate[1]) * 1e-12

    return cost