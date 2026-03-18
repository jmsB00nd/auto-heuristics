def qlosure_poly_heuristic(self, swap_gate):
    import math
    from collections import deque

    # ── Oscillation-Signature Hash-Barrier (OSHB) ──────────────────────────
    # _swap_history is a fixed-size deque populated by apply_qlosure_heuristic
    # after each *committed* swap (appendleft convention: index 0 = most recent).
    # For a candidate matching the entry at recency depth k (1 = newest),
    # multiply base cost by β^(1/k): hard wall at k=1, smooth decay for older.
    if not hasattr(self, '_swap_history'):
        self._swap_history = deque(maxlen=8)

    BETA = 16.0          # β^(1/1)=16×, β^(1/2)=4×, β^(1/3)≈2.5×, β^(1/4)=2×
    candidate = frozenset(swap_gate)
    oscillation_mult = 1.0
    for k, past_swap in enumerate(self._swap_history, start=1):
        if candidate == past_swap:
            oscillation_mult = BETA ** (1.0 / k)
            break       # shallowest (highest-penalty) match wins

    # ── Sizes ──────────────────────────────────────────────────────────────
    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # ── Anti-hotspot decay multiplier ──────────────────────────────────────
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # ── Front-layer score (dominant term) ──────────────────────────────────
    # √(deps+1) weighting honours critical-path ordering without inflating
    # values to a range where the 0.001-increment decay becomes irrelevant.
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist   = self.distance_matrix[Q1][Q2]
        deps   = math.sqrt(self.dag_dependencies_count[g] + 1)
        f_distance += deps * dist

    f_score = f_distance / front_layer_size if front_layer_size else 0.0

    # ── Extended-layer score (look-ahead, strictly subordinate) ────────────
    # W_EXT < 1 guarantees front-layer dominance (Constraint 2).
    # depth = layer_index + 1, so depth ≥ 1 always (no divide-by-zero).
    W_EXT = 0.4
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist   = self.distance_matrix[Q1][Q2]
        deps   = math.sqrt(self.dag_dependencies_count[g] + 1)
        depth  = self.extended_layer_index.get(g, 0) + 1   # ≥ 1, safe
        e_distance += deps * dist / depth

    e_score = (e_distance / extended_layer_size) if extended_layer_size else 0.0

    # ── Topology tie-breaker (deterministic, negligible magnitude) ──────────
    # Penalise swapping into low-degree "dead-end" hardware qubits.
    # Magnitude ≤ 0.1, so it only resolves ties — never overrides a real gain.
    deg0 = len(self.backend.get(swap_gate[0], [])) or 1
    deg1 = len(self.backend.get(swap_gate[1], [])) or 1
    topo_penalty = 0.05 * (1.0 / deg0 + 1.0 / deg1)

    # ── Composite cost ──────────────────────────────────────────────────────
    base_cost = max_decay * (f_score + W_EXT * e_score + topo_penalty)
    return base_cost * oscillation_mult