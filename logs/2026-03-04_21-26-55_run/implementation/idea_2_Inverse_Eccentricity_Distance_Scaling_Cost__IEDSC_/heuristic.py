def qlosure_poly_heuristic(self, swap_gate):
    # --- Eccentricity cache (computed once per routing run) ---
    # ecc[Q] = max hop-distance from Q to any other qubit in the coupling graph.
    # High eccentricity  → peripheral qubit  → amplified routing cost.
    # Low  eccentricity  → central qubit     → reduced routing cost.
    if not hasattr(self, '_iedsc_ecc'):
        n = len(self.distance_matrix)
        self._iedsc_ecc = [max(self.distance_matrix[q]) for q in range(n)]
        self._iedsc_diameter = max(self._iedsc_ecc) if self._iedsc_ecc else 1

    eccentricities = self._iedsc_ecc
    diameter       = self._iedsc_diameter

    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # Decay driven by the SWAP candidate qubits
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front-layer cost ---
    # Each gate's cost = distance(Q1,Q2) * eccentricity_scale(Q1,Q2)
    # where eccentricity_scale ∈ (0,1] is 1 for the most peripheral pair.
    # No dependency weighting: topology position, not circuit structure, drives priority.
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]

        avg_ecc   = (eccentricities[Q1] + eccentricities[Q2]) / 2.0
        ecc_scale = avg_ecc / diameter          # peripheral → scale ≈ 1

        dist    = self.distance_matrix[Q1][Q2]
        f_cost += dist * ecc_scale

    # --- Extended-layer cost ---
    # Same eccentricity scaling; layer depth discounts future gates.
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]

        layer_factor = self.extended_layer_index.get(g, 0) + 1
        avg_ecc      = (eccentricities[Q1] + eccentricities[Q2]) / 2.0
        ecc_scale    = avg_ecc / diameter

        dist    = self.distance_matrix[Q1][Q2]
        e_cost += dist * ecc_scale / layer_factor

    H = max_decay * (
        f_cost / front_layer_size
        + (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H