def qlosure_poly_heuristic(self, swap_gate):
    ALPHA = 0.5  # promotion bonus weight
    W = 1.0      # extended layer blend weight

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    if front_layer_size == 0:
        return 0.0

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front layer: dependency-weighted distance ---
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    # --- Extended layer: distance cost + promotion counting ---
    e_distance = 0.0
    promotion_count = 0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        dist = self.distance_matrix[Q1][Q2]

        e_distance += (deps + 1) * dist / layer_factor

        # Promotion condition: post-SWAP, this gate is immediately schedulable —
        # qubits are adjacent (dist == 1) AND all DAG predecessors are resolved
        if dist == 1 and deps == 0:
            promotion_count += 1

    # Base cost: decay-scaled, size-normalized distance blend
    H_dist = max_decay * (
        f_distance / front_layer_size
        + W * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    # Discrete promotion bonus: subtract normalized reward for each E→F-ready gate
    # This is the novel LPCC term — not present in any distance-only heuristic
    promotion_bonus = ALPHA * promotion_count / max(extended_layer_size, 1)

    cost = H_dist - promotion_bonus
    return cost