# Idea: Multiplicative Layer Interaction Cost (MLIC)
# Stats: {"mean_swaps": 715.1363636363636, "mean_depth": 1010.2272727272727, "mean_runtime": 4.1615774306384, "total_circuits": 22, "successful_runs": 22, "failed_runs": 0, "error": null, "first_failure_error": null, "first_failure_traceback": null}

def qlosure_poly_heuristic(self, swap_gate):
    W = 1.0
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Normalized front layer cost F
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    F = f_distance / front_layer_size  # normalized

    # Normalized extended layer cost E
    e_distance = 0.0
    if extended_layer_size:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            deps = self.dag_dependencies_count[g]
            e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor

        E = e_distance / extended_layer_size  # normalized
    else:
        E = 0.0

    # MLIC: multiplicative coupling — H = F × (1 + W×E)
    # When F → 0 (front layer nearly resolved), extended noise vanishes naturally.
    # When both F and E are large, difficulty compounds multiplicatively.
    H = max_decay * F * (1.0 + W * E)

    return H