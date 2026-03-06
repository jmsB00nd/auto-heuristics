def qlosure_poly_heuristic(self, swap_gate):
    # --- Step 1: BFS over the remaining DAG to compute Remaining Interaction Frequency ---
    # RIF(q1, q2) = total number of still-unexecuted gates acting on the logical pair
    rif = {}
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

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Step 2: Front layer — QICAC amortized cost = distance / RIF ---
    # Mathematical core: routing cost d is amortized over k future co-executions.
    # High-frequency pairs yield low amortized cost → algorithm co-locates them early
    # to eliminate repeated routing overhead for the same pair.
    # This is the INVERSE of urgency-weighting: d/k vs k*d.
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        pair = (min(q1, q2), max(q1, q2))
        k = rif.get(pair, 1)
        # Amortized cost per future interaction: penalizes leaving frequently
        # interacting pairs far apart since the overhead is paid k times.
        f_cost += self.distance_matrix[Q1][Q2] / k

    # --- Step 3: Extended lookahead — same amortization, decayed by DAG depth ---
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        pair = (min(q1, q2), max(q1, q2))
        k = rif.get(pair, 1)
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        e_cost += (self.distance_matrix[Q1][Q2] / k) / layer_factor

    W = 1.0
    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )
    return H