# Strategy: **Relay-Race Bottleneck with Harmonic Proximity Pull**
# Intuition: The front layer is a synchronization barrier — all gates must execute before the circuit advances, so routing speed is determined by the *worst-positioned* gate (the bottleneck), not the average. We minimize the **maximum** criticality-weighted distance (minimax instead of sum), coupled with a **harmonic pull** term (∝ `crit / d`) that creates a strong gravitational force toward near-executable gates without computing any before/after delta.

def qlosure_poly_heuristic(self, swap_gate):
    p0, p1 = swap_gate
    max_decay = max(self.decay_parameter[p0], self.decay_parameter[p1])

    front_layer_size = max(len(self.front_layer), 1)
    extended_layer_size = max(len(self.extended_layer), 1)

    # --- Front Layer: Minimax Bottleneck + Harmonic Proximity Pull ---
    # Relay-race insight: the circuit stalls on the worst gate, not the average.
    # Bottleneck = max(crit * d) targets that slowest runner directly.
    # Harmonic pull = sum(crit / d) / |F| rewards near-executable gates:
    #   d=1 → full crit weight; d=2 → half weight; d=3 → 1/3 weight.
    max_critical_dist = 0.0
    harmonic_pull = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        d = max(self.distance_matrix[Q1][Q2], 1)
        crit = self.dag_dependencies_count[g] + 1

        critical_dist = crit * d
        if critical_dist > max_critical_dist:
            max_critical_dist = critical_dist

        harmonic_pull += crit / d

    harmonic_pull /= front_layer_size

    # --- Extended Layer: Depth-Discounted Criticality Sum ---
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        d = max(self.distance_matrix[Q1][Q2], 1)
        depth = self.extended_layer_index.get(g, 0) + 1
        crit = self.dag_dependencies_count[g] + 1
        e_distance += crit * d / depth

    e_distance /= extended_layer_size

    # Minimize bottleneck; subtract harmonic pull (more pull = better, so negate);
    # add depth-discounted lookahead. Scaled by qubit decay.
    H = max_decay * (max_critical_dist - harmonic_pull + 0.5 * e_distance)

    return H