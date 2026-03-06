def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    if front_layer_size == 0:
        return 0.0

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Step 1: Front-layer distances (no dependency weighting) ---
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        f_distance += self.distance_matrix[Q1][Q2]

    # --- Step 2: Compute stall signal from average front-layer distance ---
    # avg_front_dist = 1  => all gates adjacent => high progress => stall = 0
    # avg_front_dist >> 1 => gates far apart    => stalling     => stall → 1
    avg_front_dist = f_distance / front_layer_size
    stall_signal = 1.0 - (1.0 / avg_front_dist)  # ∈ [0, 1)

    # --- Step 3: Derive adaptive W via feedback control ---
    # W_min: trust front layer when progress is fast
    # W_max: lean on global lookahead when routing stalls
    W_MIN, W_MAX = 0.2, 3.0
    W = W_MIN + (W_MAX - W_MIN) * stall_signal

    # --- Step 4: Extended-layer with exponential depth decay ---
    # Exponential decay 2^(-depth) falls off faster than 1/(depth+1),
    # focusing pressure on the shallowest extended gates.
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        depth = self.extended_layer_index.get(g, 0)
        e_distance += self.distance_matrix[Q1][Q2] * (2.0 ** (-depth))

    # --- Step 5: Combine with adaptive W and decay penalty ---
    H = max_decay * (
        f_distance / front_layer_size
        + W * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    return H