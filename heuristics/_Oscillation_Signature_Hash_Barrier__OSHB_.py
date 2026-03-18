def qlosure_poly_heuristic(self, swap_gate):
    from collections import deque

    # ── Constants ─────────────────────────────────────────────────────────────
    K      = 6      # rolling window: last K committed SWAPs
    LAMBDA = 3.0    # recency penalty base (>1 → exponential growth per match)
    W_EXT  = 0.5    # extended-layer weight

    # ── Lazy-init OSHB state ──────────────────────────────────────────────────
    # Survives across candidate evaluations within and between routing steps.
    # history: deque of canonical SWAP IDs, newest at index 0 (appendleft).
    if not hasattr(self, '_oshb'):
        self._oshb = {
            'history':    deque(maxlen=K),
            'prev_decay': None,
        }
    st = self._oshb

    # ── Detect last committed SWAP via decay delta ────────────────────────────
    # Invariant (routing.py lines 136, 241-242):
    #   Gate execution   → decay resets ALL qubits uniformly to 1.0
    #   Committed SWAP   → exactly 2 physical qubits gain +0.001, rest unchanged
    cur = self.decay_parameter
    if st['prev_decay'] is None:
        st['prev_decay'] = list(cur)
    else:
        prev = st['prev_decay']
        pos_changed = [q for q in range(len(cur)) if cur[q] - prev[q] >  1e-7]
        neg_changed = [q for q in range(len(cur)) if cur[q] - prev[q] < -1e-7]

        if len(pos_changed) == 2 and not neg_changed:
            # A SWAP was committed: push its canonical ID as the newest entry.
            a, b = pos_changed
            canonical_id = min(a, b) * self.num_qubits + max(a, b)
            st['history'].appendleft(canonical_id)   # newest → index 0
            st['prev_decay'] = list(cur)
        elif neg_changed:
            # Gate execution reset decay → routing context changed.
            # Old oscillation signatures are stale; start fresh.
            st['history'].clear()
            st['prev_decay'] = list(cur)

    # ── Canonical ID for the candidate SWAP ───────────────────────────────────
    q0, q1 = swap_gate
    cand_canonical = min(q0, q1) * self.num_qubits + max(q0, q1)

    # ── Positional recency penalty ─────────────────────────────────────────────
    # osc_penalty(σ) = 1 + Σ_{k=1}^{K} λ^k · 𝟙[canonical(σ) = history[k-1]]
    #
    # history[0] is the most recent committed SWAP (k=1 → λ^1 = 3).
    # history[1] is 2 steps ago (k=2 → λ^2 = 9), etc.
    #
    # Cycle detection examples:
    #   Reversing last SWAP:        k=1 match → penalty ×(1 + 3)   = 4
    #   A-B-A pattern (2-cycle):    k=1+k=2 hits → penalty ×(1+3+9) = 13
    #   A-B-C-A-B-C (3-cycle):     k=1+k=2+k=3 → penalty ×(1+3+9+27) = 40
    osc_penalty = 1.0
    for k, hist_id in enumerate(st['history'], start=1):   # k: 1 .. |history|
        if cand_canonical == hist_id:
            osc_penalty += LAMBDA ** k

    # ── Front-layer score (dominant term) ─────────────────────────────────────
    front_layer_size = max(len(self.front_layer), 1)
    f_distance = 0.0
    for g in self.front_layer:
        qa, qb  = self.access2q[g]
        Qa      = self.temp_mapping_dict[qa]
        Qb      = self.temp_mapping_dict[qb]
        deps    = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Qa][Qb]
    f_score = f_distance / front_layer_size

    # ── Extended-layer look-ahead (subordinate term) ───────────────────────────
    extended_layer_size = len(self.extended_layer)
    e_score = 0.0
    if extended_layer_size > 0:
        e_distance = 0.0
        for g in self.extended_layer:
            qa, qb       = self.access2q[g]
            Qa           = self.temp_mapping_dict[qa]
            Qb           = self.temp_mapping_dict[qb]
            layer_factor = self.extended_layer_index.get(g, 0) + 1  # ≥1, no div/0
            deps         = self.dag_dependencies_count[g]
            e_distance  += (deps + 1) * self.distance_matrix[Qa][Qb] / layer_factor
        e_score = e_distance / extended_layer_size

    # ── Anti-hotspot decay multiplier ─────────────────────────────────────────
    max_decay = max(self.decay_parameter[q0], self.decay_parameter[q1])

    # ── Composite cost: H_OSHB(σ) = osc_penalty(σ) · H_base(σ) ──────────────
    H_base = max_decay * (f_score + W_EXT * e_score)
    return osc_penalty * H_base