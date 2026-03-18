def qlosure_poly_heuristic(self, swap_gate):
    import math
    from collections import deque

    # ── Constants ────────────────────────────────────────────────────────────
    K      = 8          # trajectory window (last K committed swaps)
    BASE   = 131        # polynomial hash base (prime)
    MOD    = (1 << 31) - 1  # Mersenne prime; keeps integers bounded
    BETA   = 20.0       # oscillation barrier base
    W_EXT  = 0.35       # extended-layer weight (<1 guarantees front dominance)

    # ── Lazy-init OSHB state ─────────────────────────────────────────────────
    # State survives across candidate evaluations and routing rounds.
    # It is intentionally attached to `self` so it persists without changing
    # the class interface.
    if not hasattr(self, '_oshb'):
        self._oshb = {
            'traj':       deque(maxlen=K),  # canonical swap IDs of last K commits
            'seen':       {},               # trajectory_hash -> step last observed
            'step':       0,               # monotonic committed-swap counter
            'prev_decay': None,            # decay snapshot after last event
        }
    st = self._oshb

    # ── Committed-swap detection via decay delta ──────────────────────────────
    # Invariant (from routing.py L136, L241-242):
    #   • Gate execution:   decay resets uniformly to 1.0 for ALL qubits.
    #   • Committed SWAP:   exactly 2 physical qubits gain +0.001, rest unchanged.
    # We detect which event occurred by diffing against the previous snapshot.
    cur = self.decay_parameter
    if st['prev_decay'] is None:
        st['prev_decay'] = list(cur)
    else:
        prev = st['prev_decay']
        pos_changed = [q for q in range(len(cur)) if cur[q] - prev[q] >  1e-7]
        neg_changed = [q for q in range(len(cur)) if cur[q] - prev[q] < -1e-7]

        if len(pos_changed) == 2 and not neg_changed:
            # ── SWAP committed: update trajectory fingerprint ──────────────
            q0, q1   = pos_changed
            swap_id  = min(q0, q1) * self.num_qubits + max(q0, q1)
            st['traj'].appendleft(swap_id)          # newest at index 0

            # Recompute rolling polynomial hash over the full window O(K)
            h = 0
            for sid in st['traj']:
                h = (h * BASE + sid + 1) % MOD     # +1 avoids zero annihilation

            # Record: this trajectory hash was last seen at this step
            st['seen'][h] = st['step']
            st['step']   += 1
            st['prev_decay'] = list(cur)

        elif neg_changed:
            # ── Gate execution → decay reset: routing context changed ─────
            # Stale cycle signatures are irrelevant; start trajectory fresh.
            st['traj'].clear()
            st['seen'].clear()
            st['prev_decay'] = list(cur)

    # ── Tentative trajectory hash for this candidate ──────────────────────────
    # "What hash would the trajectory have if we committed swap_gate next?"
    q0, q1  = swap_gate
    cand_id = min(q0, q1) * self.num_qubits + max(q0, q1)

    # Tentative window: [cand_id] prepended, oldest element naturally drops
    # because the deque has maxlen=K (we simulate that here with a slice).
    tentative = [cand_id] + list(st['traj'])
    if len(tentative) > K:
        tentative = tentative[:K]

    tent_hash = 0
    for sid in tentative:
        tent_hash = (tent_hash * BASE + sid + 1) % MOD

    # ── Oscillation multiplier (cycle penalty) ────────────────────────────────
    # If tent_hash matches a previously seen trajectory fingerprint, this SWAP
    # would recreate a past routing state → we are in a swap-space cycle.
    # Penalty = BETA^(1/age): exponentially higher for recently seen signatures.
    #   age=1 → BETA^1 = 20×  (just re-entered state from 1 step ago)
    #   age=2 → BETA^0.5 ≈ 4.5×
    #   age=8 → BETA^0.125 ≈ 1.4×  (old history, gentle discouragement)
    oscillation_mult = 1.0
    if tent_hash in st['seen']:
        age              = max(st['step'] - st['seen'][tent_hash], 1)  # ≥1 always
        oscillation_mult = BETA ** (1.0 / age)

    # ── Front-layer score (dominant term, Constraint 2) ───────────────────────
    # log1p(deps) compresses the dependency range so no single gate monopolises
    # the score; still strictly monotone — higher deps = higher weight.
    front_layer_size = len(self.front_layer)
    f_distance = 0.0
    for g in self.front_layer:
        qa, qb    = self.access2q[g]
        Qa        = self.temp_mapping_dict[qa]
        Qb        = self.temp_mapping_dict[qb]
        deps      = self.dag_dependencies_count[g]
        f_distance += math.log1p(deps) * self.distance_matrix[Qa][Qb]

    f_score = f_distance / front_layer_size if front_layer_size else 0.0

    # ── Extended-layer score (subordinate look-ahead, Constraint 2) ───────────
    # sqrt(depth) attenuation: smoother falloff than 1/depth, but still ensures
    # deeper gates contribute less. W_EXT < 1 guarantees front dominance.
    extended_layer_size = len(self.extended_layer)
    e_score = 0.0
    if extended_layer_size > 0:
        e_distance = 0.0
        for g in self.extended_layer:
            qa, qb  = self.access2q[g]
            Qa      = self.temp_mapping_dict[qa]
            Qb      = self.temp_mapping_dict[qb]
            depth   = self.extended_layer_index.get(g, 0) + 1  # ≥1, no div/0
            deps    = self.dag_dependencies_count[g]
            e_distance += math.log1p(deps) * self.distance_matrix[Qa][Qb] / (depth ** 0.5)
        e_score = e_distance / extended_layer_size

    # ── Topology tie-breaker (deterministic, magnitude < 0.05) ───────────────
    # Penalise swapping into low-degree "dead-end" nodes on the hardware graph.
    # Magnitude is small enough to resolve ties only, never override real gains.
    deg0        = len(self.backend.get(q0, [])) or 1
    deg1        = len(self.backend.get(q1, [])) or 1
    topo_penalty = 0.02 * (1.0 / deg0 + 1.0 / deg1)

    # ── Anti-hotspot decay multiplier ─────────────────────────────────────────
    max_decay = max(self.decay_parameter[q0], self.decay_parameter[q1])

    # ── Composite cost ────────────────────────────────────────────────────────
    base_cost = max_decay * (f_score + W_EXT * e_score + topo_penalty)
    return base_cost * oscillation_mult