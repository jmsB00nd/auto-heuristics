def qlosure_poly_heuristic(self, swap_gate):
    W = 1.0
    p0, p1 = swap_gate[0], swap_gate[1]

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # temp_mapping_dict holds the POST-swap mapping.
    # Reconstruct PRE-swap physical location by inverting this swap.
    def pre_phys(log_q):
        post = self.temp_mapping_dict[log_q]
        if post == p0:
            return p1
        if post == p1:
            return p0
        return post

    # --- D_before: weighted distance sum using the PRE-swap mapping ---
    f_before = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        d = self.distance_matrix[pre_phys(q1)][pre_phys(q2)]
        w = self.dag_dependencies_count[g] + 1
        f_before += w * d

    e_before = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        d = self.distance_matrix[pre_phys(q1)][pre_phys(q2)]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        w = self.dag_dependencies_count[g] + 1
        e_before += w * d / layer_factor

    D_before = (f_before / front_layer_size) + W * (
        (e_before / extended_layer_size) if extended_layer_size else 0.0
    )

    # --- D_after: weighted distance sum using the POST-swap mapping ---
    f_after = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        d = self.distance_matrix[self.temp_mapping_dict[q1]][self.temp_mapping_dict[q2]]
        w = self.dag_dependencies_count[g] + 1
        f_after += w * d

    e_after = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        d = self.distance_matrix[self.temp_mapping_dict[q1]][self.temp_mapping_dict[q2]]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        w = self.dag_dependencies_count[g] + 1
        e_after += w * d / layer_factor

    D_after = (f_after / front_layer_size) + W * (
        (e_after / extended_layer_size) if extended_layer_size else 0.0
    )

    # MDRC: algebraic change in total weighted distance.
    # Negative = SWAP reduces distance = good.
    # The scheduler selects the minimum H, so most-negative wins.
    cost = D_after - D_before

    return cost