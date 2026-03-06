def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    if front_layer_size == 0:
        return 0.0

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front layer: urgency = (fan-out + 1)^2 ---
    # Fan-out = direct successors in the 2q DAG: gates immediately unlocked upon completion.
    # Quadratic scaling captures the super-linear impact of bottleneck-breaking gates.
    f_cost = 0.0
    f_urgency_total = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]

        fanout = len(self.dag2q.get(g, set()))
        urgency = (fanout + 1) ** 2   # quadratic: bottleneck gates are disproportionately critical

        f_cost += urgency * dist
        f_urgency_total += urgency

    # Normalize by total URGENCY MASS, not layer count.
    # This biases the score toward the routing difficulty of the most bottleneck-breaking gate.
    f_normalized = f_cost / f_urgency_total

    # --- Extended layer: fan-out urgency discounted by depth^2 ---
    # Deeper gates are less immediately relevant; depth^2 falloff is steeper than baseline's 1/depth.
    e_cost = 0.0
    e_urgency_total = 0.0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]

        layer_idx = self.extended_layer_index.get(g, 0) + 1
        fanout = len(self.dag2q.get(g, set()))
        urgency = (fanout + 1) / (layer_idx ** 2)

        e_cost += urgency * dist
        e_urgency_total += urgency

    e_normalized = (e_cost / e_urgency_total) if e_urgency_total > 0 else 0.0

    # --- Combine via harmonic mean ---
    # Penalizes swaps that resolve one layer well but leave the other poorly addressed.
    # Fundamentally different from baseline's additive linear combination.
    if extended_layer_size > 0 and (f_normalized + e_normalized) > 0:
        combined = 2.0 * f_normalized * e_normalized / (f_normalized + e_normalized)
    else:
        combined = f_normalized

    return max_decay * combined