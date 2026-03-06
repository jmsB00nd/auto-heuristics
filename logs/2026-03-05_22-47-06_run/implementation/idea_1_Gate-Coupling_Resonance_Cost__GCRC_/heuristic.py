def qlosure_poly_heuristic(self, swap_gate):
    # ── Gate-Coupling Resonance Cost (GCRC) ──────────────────────────────────
    # Pair-centric globally-informed routing cost.
    # For each unique logical pair (q1,q2) active in F∪E, the resonance cost
    # = (rif[pair] / total_remaining) × distance × urgency,
    # where RIF is computed over the *entire* remaining circuit via BFS,
    # making every routing decision globally aware of future coupling demand.

    # ── Step 1: BFS over full remaining DAG → RIF per logical pair ───────────
    rif = {}       # (min_q, max_q) → count of remaining interactions
    visited = set()
    stack = list(self.front_layer)
    while stack:
        gate = stack.pop()
        if gate in visited:
            continue
        visited.add(gate)
        q1, q2 = self.access2q[gate]
        pair = (min(q1, q2), max(q1, q2))
        rif[pair] = rif.get(pair, 0) + 1
        for succ in self.dag2q.get(gate, set()):
            if succ not in visited:
                stack.append(succ)

    if not rif:
        return 0.0

    # Normalize by total remaining 2q work so each pair contributes its
    # *fractional workload share* — a circuit-global resonance weight.
    total_remaining = sum(rif.values())

    # ── Step 2: Urgency tiers from F∪E membership + lookahead depth decay ────
    # Front-layer pairs have urgency 1.0; extended-layer pairs decay as 1/depth.
    # Pairs with multiple active gates keep their maximum urgency.
    urgency = {}
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        pair = (min(q1, q2), max(q1, q2))
        urgency[pair] = max(urgency.get(pair, 0.0), 1.0)

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        pair = (min(q1, q2), max(q1, q2))
        depth = self.extended_layer_index.get(g, 0) + 1
        urgency[pair] = max(urgency.get(pair, 0.0), 1.0 / depth)

    max_decay = max(self.decay_parameter[swap_gate[0]],
                    self.decay_parameter[swap_gate[1]])

    # ── Step 3: Resonance-weighted cost over all active pairs ─────────────────
    # cost(pair) = (global_frequency_share) × physical_distance × urgency
    # Pairs that dominate future circuit work AND are physically distant
    # AND are imminent (high urgency) drive the routing decision.
    cost = 0.0
    for pair, u in urgency.items():
        q1, q2 = pair
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        freq_weight = rif[pair] / total_remaining   # fraction of remaining work
        cost += freq_weight * self.distance_matrix[Q1][Q2] * u

    return max_decay * cost