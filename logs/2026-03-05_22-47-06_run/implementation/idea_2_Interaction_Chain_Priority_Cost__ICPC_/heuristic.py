def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    if front_layer_size == 0:
        return 0

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # ── Build: logical qubit → [(extended_gate, layer_index, dist)] ──────────
    # This captures the "interaction chain" downstream of each qubit.
    qubit_chain = {}
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        eQ1, eQ2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        future_dist = self.distance_matrix[eQ1][eQ2]
        layer_idx = self.extended_layer_index.get(g, 0)
        for q in (q1, q2):
            if q not in qubit_chain:
                qubit_chain[q] = []
            qubit_chain[q].append((future_dist, layer_idx))

    # ── Front-layer cost ──────────────────────────────────────────────────────
    # For each gate g(q1,q2) in F, compute how much routing g *now* will
    # perturb the downstream interaction chain through q1 and q2.
    # chain_impact = Σ  future_dist(e) / (layer_index(e) + 1)
    #               e∈E sharing a qubit with g
    # Gate cost = current_distance * (1 + chain_impact)
    # → gates whose qubits anchor deep successor chains are penalised more
    #   when they are far away, strongly incentivising the router to close
    #   them first.
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        base_dist = self.distance_matrix[Q1][Q2]

        chain_impact = 0.0
        for q in (q1, q2):
            for future_dist, layer_idx in qubit_chain.get(q, []):
                # Earlier gates in the chain contribute more urgency
                chain_impact += future_dist / (layer_idx + 1)

        f_cost += base_dist * (1.0 + chain_impact)

    # ── Extended-layer cost ───────────────────────────────────────────────────
    # Standard look-ahead: distance decayed by how deep in the extended layer
    # the gate sits.  No dependency reweighting — that is handled above via
    # chain_impact on the front layer.
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        e_cost += self.distance_matrix[Q1][Q2] / layer_factor

    H = max_decay * (
        f_cost / front_layer_size
        + (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H