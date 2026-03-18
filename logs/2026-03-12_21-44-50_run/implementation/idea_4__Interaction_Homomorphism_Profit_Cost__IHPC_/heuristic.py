def qlosure_poly_heuristic(self, swap_gate):
    W = 0.5        # extended-layer weight
    lambda_ = 1.5  # profit amplification ∈ [0.5, 2.0]

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front layer: distance cost + homomorphism profit ---
    f_distance  = 0.0
    resolved_F  = 0.0
    total_F     = 0.0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps   = self.dag_dependencies_count[g]
        weight = deps + 1

        f_distance += weight * self.distance_matrix[Q1][Q2]
        total_F    += weight

        if (Q1, Q2) in self.backend_connections or (Q2, Q1) in self.backend_connections:
            resolved_F += weight

    # --- Extended layer: homomorphism profit only (discounted by layer depth) ---
    resolved_E = 0.0
    total_E    = 0.0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps   = self.dag_dependencies_count[g]
        weight = (deps + 1) / layer_factor

        total_E += weight

        if (Q1, Q2) in self.backend_connections or (Q2, Q1) in self.backend_connections:
            resolved_E += weight

    # homomorphism_profit ∈ [0, 1+W] — fully bounded, no div-by-zero
    homomorphism_profit = (resolved_F / max(total_F, 1.0)) + W * (resolved_E / max(total_E, 1.0))

    # Minimise: distance cost − profit bonus
    # Subtracting profit makes swaps that unlock more adjacencies strictly cheaper
    H = max_decay * (
        f_distance / front_layer_size
        - lambda_ * homomorphism_profit
    )

    return H