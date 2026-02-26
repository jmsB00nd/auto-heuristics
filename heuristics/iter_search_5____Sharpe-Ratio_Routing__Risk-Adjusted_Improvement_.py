# Strategy: ** Sharpe-Ratio Routing (Risk-Adjusted Improvement)
# Intuition: ** Borrowing from portfolio theory: a SWAP should maximize the *risk-adjusted* criticality-weighted improvement across all gates. Two SWAPs with identical mean improvement should prefer the lower-variance one — because quantum routing is a synchronization barrier, not an average, so high variance (some gates improve greatly, others worsen) is genuinely riskier than uniform progress. We score via `H = decay * (−mean_improvement + λ·std_improvement)`.

def qlosure_poly_heuristic(self, swap_gate):
    p0, p1 = swap_gate

    # Reconstruct pre-swap physical location by inverting the swap
    def pre_phys(log_q):
        post = self.temp_mapping_dict[log_q]
        if post == p0: return p1
        if post == p1: return p0
        return post

    improvements = []

    # Front layer: full weight — synchronization barrier, highest priority
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        d_pre = self.distance_matrix[pre_phys(q1)][pre_phys(q2)]
        d_post = self.distance_matrix[self.temp_mapping_dict[q1]][self.temp_mapping_dict[q2]]
        crit = self.dag_dependencies_count[g] + 1
        improvements.append((d_pre - d_post) * crit)  # positive = good

    # Extended lookahead: discounted by depth
    W_ext = 0.5
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        d_pre = self.distance_matrix[pre_phys(q1)][pre_phys(q2)]
        d_post = self.distance_matrix[self.temp_mapping_dict[q1]][self.temp_mapping_dict[q2]]
        depth = self.extended_layer_index.get(g, 0) + 1
        crit = self.dag_dependencies_count[g] + 1
        improvements.append(W_ext * (d_pre - d_post) * crit / depth)

    if not improvements:
        return 0.0

    n = len(improvements)
    mean_imp = sum(improvements) / n
    variance = sum((x - mean_imp) ** 2 for x in improvements) / n
    std_imp = variance ** 0.5

    # Qubit thermal penalty
    decay_cost = max(self.decay_parameter[p0], self.decay_parameter[p1])

    # Sharpe-inspired: maximize mean, minimize variance
    # Lower H = better SWAP choice
    lambda_risk = 0.5
    H = decay_cost * (-mean_imp + lambda_risk * std_imp)

    return H