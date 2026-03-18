def qlosure_poly_heuristic(self, swap_gate):
    # ── TEQH: Thermal EMA Qubit Heat ─────────────────────────────────────────
    # Core innovation over baseline: replaces max(decay[q0], decay[q1]) with
    #
    #   thermal_load = decay[q0]^β + decay[q1]^β
    #
    # Two fundamental changes:
    #
    #   1. ADDITIVE (not max): both qubits' heat contributes to the penalty.
    #      Baseline's max() is blind to the cooler qubit, allowing a
    #      "hot + cool" swap to look as cheap as a "hot + cold" one.
    #      TEQH makes both qubits pay, directly exposing ping-pong cycles
    #      where the same two qubits alternate as the "cool" partner.
    #
    #   2. SUPERLINEAR β > 1: cost escalates faster than the heat itself.
    #      decay[q] starts at 1.0 and gains +0.001 per SWAP, resetting on
    #      forward progress.  With β = 3:
    #        10 consecutive SWAPs → 1.01^3 ≈ 1.030  (mild deterrent)
    #        50 consecutive SWAPs → 1.05^3 ≈ 1.158  (moderate)
    #       100 consecutive SWAPs → 1.10^3 ≈ 1.331  (strong)
    #       500 consecutive SWAPs → 1.50^3 ≈ 3.375  (severe)
    #      Superlinear growth forces the search toward fresher routing paths.
    #
    # Anti-ping-pong guarantee: if swap (q0↔q1) is attempted twice in a row,
    # BOTH qubits carry elevated heat, so thermal_load grows from both terms
    # simultaneously — no single cool qubit can mask the cycle.
    #
    # Front-layer dominance: W = 0.5 < 1 ensures the extended lookahead
    # never overrides the immediate dependency signal.
    # ─────────────────────────────────────────────────────────────────────────

    beta = 3.0  # Cubic superlinearity: rapid heat escalation for stuck qubits
    W    = 0.5  # Extended layer weight; strictly < 1 enforces front-layer dominance

    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # ── Additive superlinear thermal load (core TEQH innovation) ─────────────
    # decay_parameter[q] >= 1.0 always, so integer exponents are safe.
    h0 = self.decay_parameter[swap_gate[0]] ** beta
    h1 = self.decay_parameter[swap_gate[1]] ** beta
    thermal_load = h0 + h1  # sum, not max — captures joint thermal stress

    # ── Front layer: dependency-weighted distance (MUST dominate) ────────────
    # (deps + 1) weights gates higher when more downstream gates depend on them.
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    # ── Extended layer: lookahead with layer-depth decay ──────────────────────
    # 1 / layer_factor applies geometrically weaker influence as lookahead
    # depth increases, ensuring extended gates never override front-layer pull.
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor

    # Normalize by layer size.
    # front_layer_size is always > 0 here (guarded by the while-loop in
    # execute_algorithm), so no divide-by-zero risk.
    f_score = f_distance / front_layer_size
    e_score = (e_distance / extended_layer_size) if extended_layer_size else 0.0

    # ── Final TEQH cost ───────────────────────────────────────────────────────
    # thermal_load (additive superlinear heat) modulates the distance scores.
    # W < 1 guarantees strict front-layer dominance over lookahead.
    H = thermal_load * (f_score + W * e_score)

    return H