def qlosure_poly_heuristic(self, swap_gate):
    W = 1.0
    BETA = 0.6  # divergence penalty weight: second-order signal strength

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    p1, p2 = swap_gate[0], swap_gate[1]
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    def pre_swap_physical(q):
        """
        Reconstruct the physical position of logical qubit q BEFORE the swap.
        temp_mapping_dict reflects post-swap state:
          - if q is now at p1, it was at p2 before (and vice versa)
          - all other qubits are unaffected
        """
        phys = self.temp_mapping_dict[q]
        if phys == p1:
            return p2
        elif phys == p2:
            return p1
        return phys

    f_distance = 0.0
    divergence_penalty = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        deps = self.dag_dependencies_count[g]
        weight = deps + 1

        # Distance after applying the candidate swap
        Q1_new = self.temp_mapping_dict[q1]
        Q2_new = self.temp_mapping_dict[q2]
        dist_after = self.distance_matrix[Q1_new][Q2_new]

        # Distance before applying the candidate swap (reverse the swap's effect)
        Q1_old = pre_swap_physical(q1)
        Q2_old = pre_swap_physical(q2)
        dist_before = self.distance_matrix[Q1_old][Q2_old]

        # Velocity: positive => qubits diverging (bad routing momentum)
        velocity = dist_after - dist_before
        divergence = max(0.0, velocity)

        f_distance += weight * dist_after
        divergence_penalty += weight * divergence

    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor

    # DVC cost: base distance + divergence penalty (second-order routing signal)
    H = max_decay * (
        (f_distance + BETA * divergence_penalty) / front_layer_size
        + W * ((e_distance / extended_layer_size) if extended_layer_size else 0)
    )

    return H