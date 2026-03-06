# Idea: Circuit Phase Adaptive Lookahead Cost (CPALC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on multiplier_n75__1308CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    W     = 1.0   # base lookahead weight
    alpha = 2.0   # exponential decay rate

    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    # ── 1. Circuit progress: count executed 2q-gates ──────────────────────
    # BFS forward from front_layer to find ALL remaining (unexecuted) gates
    T_total = len(self.access2q)

    visited = set(self.front_layer)
    queue   = list(self.front_layer)
    while queue:
        g = queue.pop()
        for succ in self.dag2q.get(g, set()):
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    t        = max(0, T_total - len(visited))   # gates already executed
    progress = t / T_total if T_total > 0 else 0.0

    # ── 2. Adaptive lookahead weight ───────────────────────────────────────
    # W_eff → W at start (t=0), W_eff → 0 near end (t≈T_total)
    W_eff = W * math.exp(-alpha * progress)

    # ── 3. Hardware noise factor ───────────────────────────────────────────
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # ── 4. Front-layer cost: dependency-weighted distance ─────────────────
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps       = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    # ── 5. Extended-layer cost: depth-discounted, dependency-weighted ──────
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps         = self.dag_dependencies_count[g]
        e_distance  += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor

    # ── 6. Combine: adaptive W_eff smoothly silences lookahead over time ───
    H = max_decay * (
        f_distance / front_layer_size
        + W_eff * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    return H