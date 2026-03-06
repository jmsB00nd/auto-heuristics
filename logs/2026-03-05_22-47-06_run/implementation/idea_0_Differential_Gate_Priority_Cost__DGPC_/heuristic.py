def qlosure_poly_heuristic(self, swap_gate):
    alpha = 2.5  # Regressive amplification factor (> 1) — penalizes any distance regression
    W = 0.5      # Extended layer relative weight

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    p1, p2 = swap_gate[0], swap_gate[1]
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    def phys_after(Q):
        """Simulate physical qubit position after applying the candidate SWAP."""
        if Q == p1: return p2
        if Q == p2: return p1
        return Q

    def gate_asymmetric_cost(g, layer_factor=1.0):
        """
        Compute the asymmetric differential cost for a single gate g.

        ΔD(g) = d_after − d_before
          - ΔD ≤ 0 (improvement): normal weighted distance d_after
          - ΔD > 0 (regression): extra regressive penalty → d_after + (alpha−1)·ΔD
                                = d_before + alpha·ΔD
        This is equivalent to:
            cost = weight · (d_after + (alpha − 1) · max(0, ΔD))
        """
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]

        d_before = self.distance_matrix[Q1][Q2]
        d_after  = self.distance_matrix[phys_after(Q1)][phys_after(Q2)]

        delta_d = d_after - d_before                        # signed differential
        regression_penalty = (alpha - 1) * max(0.0, delta_d)  # zero when improving

        weight = (self.dag_dependencies_count[g] + 1) / layer_factor
        return weight * (d_after + regression_penalty)

    # --- Front layer: full priority, no layer discounting ---
    f_cost = sum(gate_asymmetric_cost(g) for g in self.front_layer)

    # --- Extended layer: discounted by layer depth ---
    e_cost = 0.0
    for g in self.extended_layer:
        lf = self.extended_layer_index.get(g, 0) + 1
        e_cost += gate_asymmetric_cost(g, layer_factor=lf)

    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0)
    )

    return H