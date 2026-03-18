def qlosure_poly_heuristic(self, swap_gate):
    """
    Stagnation-Adaptive Front Layer Pressure (SAFLP)
    =================================================
    Core idea: extended-layer weight W is NOT fixed. It is suppressed
    continuously as stagnation accumulates, forcing the cost function to
    become a pure front-layer minimiser exactly when routing gets stuck.

    Stagnation proxy (state-free, no extra bookkeeping):
        decay resets to 1.0 on every gate execution (execute_algorithm L146).
        Each applied SWAP adds +0.001 to exactly two physical qubits (L241-242).
        Therefore:  stagnation_count = round(sum(max(d-1,0) for d in decay) / 0.002)
        This equals the number of consecutive SWAPs since the last gate fired.

    Adaptive weight  W(s) = W_max / (1 + alpha * s)
        s = 0  → W = 1.0  (no stagnation: balanced lookahead enabled)
        s → ∞  → W → 0    (full stagnation: pure front-layer pressure)
        Denominator >= 1 always → no divide-by-zero possible.

    Anti-cycling (differs from baseline):
        decay_score = (d0 + d1) / 2  (average heat, NOT max)
        Penalises cumulative qubit activity: a ping-pong pair where BOTH
        qubits are hot earns a higher penalty than one hot + one cold qubit,
        which max_decay cannot distinguish.

    Complexity: O(|FL| + |EL| + N_phys) per call — all linear sweeps, no
    exotic data structures, no floating-point edge cases.
    """
    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    d0 = self.decay_parameter[swap_gate[0]]
    d1 = self.decay_parameter[swap_gate[1]]

    # Average heat penalises cumulative pair activity (anti-cycling signal)
    decay_score = (d0 + d1) / 2.0

    # ── Front Layer Score ────────────────────────────────────────────────────
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2  = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps    = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    f_score = f_distance / front_layer_size          # front_layer always non-empty here

    # ── Extended Layer Score ─────────────────────────────────────────────────
    e_score = 0.0
    if extended_layer_size > 0:
        e_distance = 0.0
        for g in self.extended_layer:
            q1, q2      = self.access2q[g]
            Q1, Q2      = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            layer_factor = self.extended_layer_index.get(g, 0) + 1
            deps         = self.dag_dependencies_count[g]
            e_distance  += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor
        e_score = e_distance / extended_layer_size

    # ── Stagnation-Adaptive Weight ────────────────────────────────────────────
    # Recover stagnation count from accumulated decay — zero external state.
    total_decay_excess = sum(max(d - 1.0, 0.0) for d in self.decay_parameter)
    stagnation_count   = round(total_decay_excess / 0.002)

    # W_max / (1 + alpha*s)  ∈ (0, W_max]:  always positive, never zero.
    W_max    = 1.0
    alpha    = 0.15                                          # sensitivity: ~halved at s≈7
    W_adaptive = W_max / (1.0 + alpha * stagnation_count)

    H = decay_score * (f_score + W_adaptive * e_score)

    return H