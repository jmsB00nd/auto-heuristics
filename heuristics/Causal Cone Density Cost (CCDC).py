def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    if front_layer_size == 0:
        return 0.0

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Normalize cone sizes by the largest cone in the front layer ---
    # dag_dependencies_count[g] = |transitive successor set of g| = causal cone size
    front_cones = [self.dag_dependencies_count[g] for g in self.front_layer]
    max_front_cone = max(front_cones) if front_cones else 1

    # --- Front Layer: Quadratic Causal Cone Density Weighting ---
    # rho(g) in [0,1]: fraction of future circuit controlled by gate g
    # Hub gates (rho -> 1) get weight (1 + rho^2) -> up to 2x; non-hubs (rho -> 0) get weight ~1
    # This is superlinear in cone breadth, unlike baseline's linear (deps+1)
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        rho = self.dag_dependencies_count[g] / max_front_cone
        hub_weight = 1.0 + rho * rho  # quadratic amplification of causal breadth
        f_cost += hub_weight * self.distance_matrix[Q1][Q2]

    f_cost /= front_layer_size

    # --- Extended Layer: Cone-Modulated Depth Decay ---
    # For extended gates, depth penalty = layer_idx / (1 + rho):
    #   - Low-cone gate at depth 3 → penalty = 3/(1+0) = 3.0  (strongly discounted)
    #   - Hub gate at depth 3      → penalty = 3/(1+1) = 1.5  (resists discounting)
    # Gates with high causal cone breadth remain relevant even deep in the lookahead
    e_cost = 0.0
    if extended_layer_size > 0:
        ext_cones = [self.dag_dependencies_count[g] for g in self.extended_layer]
        max_ext_cone = max(ext_cones) if ext_cones else 1

        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            rho = self.dag_dependencies_count[g] / max_ext_cone
            layer_idx = self.extended_layer_index.get(g, 0) + 1
            # Cone density attenuates depth discount: hubs stay influential at depth
            effective_depth_penalty = layer_idx / (1.0 + rho)
            e_cost += self.distance_matrix[Q1][Q2] / effective_depth_penalty

        e_cost /= extended_layer_size

    W = 0.5
    H = max_decay * (f_cost + W * e_cost)
    return H