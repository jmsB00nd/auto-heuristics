def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = max(len(self.front_layer), 1)
    extended_layer_size = max(len(self.extended_layer), 1)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front layer: criticality-weighted physical distance ---
    f_distance = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    # --- Extended layer: Gate Execution Proximity weighting ---
    # exec_proximity(g) = |predecessors(g) ∩ front_layer| / |predecessors(g)|
    # Replaces static 1/layer_factor with a dynamic DAG-nearness signal:
    # the more of g's direct predecessors are already in the front layer,
    # the sooner g will become executable → heavier influence on routing.
    e_distance = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]

        preds = self.dag_predecessors2q.get(g, set())
        num_preds = len(preds)

        if num_preds == 0:
            # No 2q predecessors → gate is structurally near-ready; treat as fully proximate
            exec_proximity = 1.0
        else:
            preds_in_front = len(preds & self.front_layer)
            exec_proximity = preds_in_front / num_preds

        deps = self.dag_dependencies_count[g]
        e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] * exec_proximity

    H = max_decay * (
        f_distance / front_layer_size
        + (e_distance / extended_layer_size if extended_layer_size else 0)
    )

    return H