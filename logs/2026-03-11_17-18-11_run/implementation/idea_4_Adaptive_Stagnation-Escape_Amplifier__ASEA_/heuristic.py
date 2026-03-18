def qlosure_poly_heuristic(self, swap_gate):
    """
    Adaptive Stagnation-Escape Amplifier (ASEA) heuristic.

    Tracks the best (minimum) normalised front-layer distance achievable by
    any candidate swap in each routing round.  When that minimum fails to
    decrease for K consecutive rounds the front-layer weight W_front is
    amplified exponentially (2^excess), forcing the optimiser into aggressive
    front-layer mode and breaking the local optimum.  As soon as improvement
    resumes the amplifier resets to 1.0.

    Round boundaries are detected in O(1) by observing that
    `self.extended_layer` is a freshly allocated object at the start of
    every call to `apply_qlosure_heuristic`, so its CPython identity
    (`id`) changes every round without any extra bookkeeping.
    """

    # ── Lazy-init of persistent ASEA state ───────────────────────────────
    # These attributes survive across calls within a single routing session.
    if not hasattr(self, '_asea_stagnation_count'):
        self._asea_stagnation_count  = 0
        self._asea_amplifier         = 1.0   # W_front multiplier (≥ 1)
        self._asea_round_min_f       = float('inf')  # best f_score seen THIS round
        self._asea_prev_round_min_f  = float('inf')  # best f_score from PREVIOUS round
        self._asea_round_token       = None   # id(extended_layer) at round start
        self._asea_K                 = 5      # stagnation threshold (rounds)
        self._asea_W                 = 0.5    # extended-layer weight, fixed < 1

    # ── Round-boundary detection & stagnation update ─────────────────────
    # `extended_layer` is reconstructed at the top of every
    # `apply_qlosure_heuristic` call, guaranteeing a new CPython id.
    current_token = id(self.extended_layer)
    if current_token != self._asea_round_token:
        # --- Evaluate stagnation from the just-finished round ---
        if self._asea_round_token is not None:   # skip on the very first round
            improved = self._asea_round_min_f < self._asea_prev_round_min_f - 1e-9
            if improved:
                # Genuine progress: reset amplifier and counter
                self._asea_stagnation_count = 0
                self._asea_amplifier        = 1.0
            else:
                # No progress: escalate
                self._asea_stagnation_count += 1
                if self._asea_stagnation_count >= self._asea_K:
                    # Exponential amplification, capped to preserve numerical
                    # stability and keep decay_parameter effective as tie-breaker
                    excess = self._asea_stagnation_count - self._asea_K + 1
                    self._asea_amplifier = min(2.0 ** excess, 32.0)

        # Advance round bookkeeping
        self._asea_prev_round_min_f = self._asea_round_min_f
        self._asea_round_min_f      = float('inf')
        self._asea_round_token      = current_token

    # ── Front-layer score ─────────────────────────────────────────────────
    # Each gate is weighted by (sqrt(deps) + 1) * distance.
    # sqrt compresses the dependency scale so that the 0.001 per-swap decay
    # increment remains a meaningful tie-breaker even on deep circuits.
    front_layer_size = len(self.front_layer)
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2   = self.access2q[g]
        Q1, Q2   = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps     = math.sqrt(self.dag_dependencies_count[g])
        f_distance += (deps + 1.0) * self.distance_matrix[Q1][Q2]

    f_score = f_distance / front_layer_size if front_layer_size else 0.0

    # Track the best achievable f_score across all candidates this round.
    # Computed BEFORE amplification so the stagnation detector stays unbiased.
    if f_score < self._asea_round_min_f:
        self._asea_round_min_f = f_score

    # ── Extended-layer score ──────────────────────────────────────────────
    # Gates deeper in the look-ahead horizon contribute less (1/layer_factor).
    extended_layer_size = len(self.extended_layer)
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2        = self.access2q[g]
        Q1, Q2        = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor  = self.extended_layer_index.get(g, 0) + 1
        deps          = math.sqrt(self.dag_dependencies_count[g])
        e_distance   += (deps + 1.0) * self.distance_matrix[Q1][Q2] / layer_factor

    e_score = e_distance / extended_layer_size if extended_layer_size else 0.0

    # ── Final cost ────────────────────────────────────────────────────────
    # W_front  ≥ 1 always, so front-layer ALWAYS dominates extended-layer.
    # During stagnation W_front escalates as 2^k (k = excess stagnation rounds),
    # widening that gap and forcing aggressive front-layer reduction.
    # max_decay penalises repeatedly swapping the same physical qubit pair,
    # acting as the primary anti-ping-pong / tie-breaking mechanism.
    max_decay = max(self.decay_parameter[swap_gate[0]],
                    self.decay_parameter[swap_gate[1]])

    W_front = self._asea_amplifier   # ≥ 1.0, escalates during stagnation
    W       = self._asea_W           # 0.5,   fixed extended-layer weight

    H = max_decay * (W_front * f_score + W * e_score)
    return H