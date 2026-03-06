def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # --- Circuit completion fraction p ∈ [0, 1] ---
    # BFS from front layer counts all remaining (unexecuted) 2q gates
    total_2q = len(self.dag_dependencies_count)
    visited = set()
    stack = list(self.front_layer)
    while stack:
        gate = stack.pop()
        if gate in visited:
            continue
        visited.add(gate)
        for succ in self.dag2q.get(gate, set()):
            if succ not in visited:
                stack.append(succ)
    remaining_2q = len(visited)
    p = 1.0 - (remaining_2q / total_2q) if total_2q > 0 else 0.0
    p = min(1.0, max(0.0, p))

    # Quadratic decay: W=1 at circuit start (p=0), W=0 at circuit end (p=1)
    # Extended lookahead is progressively suppressed as routing completes
    W = (1.0 - p) ** 2

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor

    H = max_decay * (
        f_distance / front_layer_size +
        W * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    return H