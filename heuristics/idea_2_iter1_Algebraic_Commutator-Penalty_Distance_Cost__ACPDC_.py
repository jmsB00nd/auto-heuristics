# Idea: Algebraic Commutator-Penalty Distance Cost (ACPDC)
# Stats: {"mean_swaps": 681.8181818181819, "mean_depth": 1077.1363636363637, "mean_runtime": 4.6877371723001655, "total_circuits": 22, "successful_runs": 22, "failed_runs": 0, "error": null, "first_failure_error": null, "first_failure_traceback": null}

def qlosure_poly_heuristic(self, swap_gate):
    alpha = 0.5  # commutator-penalty scaling factor
    W = 1.0
    front_layer_size = max(len(self.front_layer), 1)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    def compute_C_g(gate_id):
        """
        Count non-commuting successors of gate_id that share at least one
        logical qubit with it. Two gates sharing a qubit in a serialized DAG
        impose a strict ordering constraint (algebraic commutator != 0).
        """
        gate_qubits = set(self.access2q[gate_id])
        successors = self.dag2q.get(gate_id, set())
        count = 0
        for succ in successors:
            succ_qubits = set(self.access2q[succ])
            if gate_qubits & succ_qubits:  # shared qubit => non-commuting
                count += 1
        return count

    f_distance = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        C_g = compute_C_g(g)
        # (1 + alpha * C_g): gates deeper in a non-commuting chain are more urgent
        weight = 1.0 + alpha * C_g
        f_distance += weight * self.distance_matrix[Q1][Q2]

    e_distance = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        C_g = compute_C_g(g)
        weight = 1.0 + alpha * C_g
        e_distance += weight * self.distance_matrix[Q1][Q2] / layer_factor

    H = max_decay * (
        f_distance / front_layer_size
        + W * (e_distance / extended_layer_size if extended_layer_size else 0)
    )

    return H