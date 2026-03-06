# Idea: Commutator-Aware Reordering Window Cost (CARWC)
# Stats: {"mean_swaps": 1057.8181818181818, "mean_depth": 1056.3636363636363, "mean_runtime": 1.9244695793498645, "total_circuits": 22, "successful_runs": 22, "failed_runs": 0, "error": null, "first_failure_error": null, "first_failure_traceback": null}

def qlosure_poly_heuristic(self, swap_gate):
    W = 3  # commutation lookahead window depth

    # Two 2-qubit gates commute if they act on completely disjoint qubit sets
    def gates_commute(g1, g2):
        qa, qb = self.access2q[g1]
        qc, qd = self.access2q[g2]
        return not ({qa, qb} & {qc, qd})

    # BFS: build commutativity-expanded effective front.
    # A successor joins the expanded front if it commutes with its blocking
    # predecessor — meaning it can be reordered before that predecessor.
    expanded_front = {g: 0 for g in self.front_layer}
    bfs_queue = list(self.front_layer)
    head = 0

    while head < len(bfs_queue):
        g = bfs_queue[head]
        head += 1
        d = expanded_front[g]
        if d >= W:
            continue
        for succ in self.dag2q.get(g, set()):
            if succ in expanded_front:
                continue
            # succ commutes with g => can be executed before g (commutation swap)
            if gates_commute(g, succ):
                expanded_front[succ] = d + 1
                bfs_queue.append(succ)

    # Stuckness of a front-layer gate g:
    #   fraction of g's successors that cannot commute past g.
    # stuckness=1.0 → g is a genuine serialisation bottleneck (must be resolved).
    # stuckness=0.0 → all successors can bypass g; this gate is less urgent.
    def stuckness(g):
        succs = self.dag2q.get(g, set())
        if not succs:
            return 1.0  # no successors to check, treat as a hard barrier
        blocked = sum(1 for s in succs if not gates_commute(g, s))
        return blocked / len(succs)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Front-layer cost ---
    # Stuck gates receive a high stuckness weight (up to 2.0) because they
    # form genuine serialisation barriers — only a real SWAP can unblock them.
    # Commutable gates receive a low weight (down to 0.5) because their
    # successors can be reordered around them, so phantom SWAP pressure is
    # removed.
    front_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        s = stuckness(g)
        # Weight ∈ [0.5, 2.0]: stuck → 2.0, fully commutable → 0.5
        w = 0.5 + 1.5 * s
        front_cost += w * (deps + 1) * self.distance_matrix[Q1][Q2]

    # --- Commutation-expansion cost ---
    # Gates newly admitted to the effective front via commutation reordering.
    # They could execute sooner than the static DAG order implies; proximity
    # to the actual front (small depth d) makes them more immediately valuable.
    expansion_cost = 0.0
    n_expansion = 0
    for g, d in expanded_front.items():
        if g in self.front_layer:
            continue
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        # Depth discount: gates one commutation step away matter more
        layer_factor = d + 1
        expansion_cost += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor
        n_expansion += 1

    fl_size  = max(len(self.front_layer), 1)
    exp_size = max(n_expansion, 1)

    # Combine: front dominates (weight 1.0), expansion discounted (weight 0.5)
    # so that phantom costs from commutable gate pairs are genuinely suppressed.
    H = max_decay * (
        front_cost  / fl_size
        + 0.5 * expansion_cost / exp_size
    )

    return H