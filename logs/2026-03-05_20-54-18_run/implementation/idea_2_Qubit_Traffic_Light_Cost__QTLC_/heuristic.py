def qlosure_poly_heuristic(self, swap_gate):
    """
    Qubit Traffic Light Cost (QTLC)

    Maintains an idle_time counter per logical qubit: the number of routing
    steps since that qubit last participated in a completed gate.

    Intuition:
      - Gates whose logical qubits have been "starved" (idle for many steps)
        accumulate urgency: urgency(q) = 1 + log(1 + idle_time[q])
      - The total cost sums urgency-weighted distances across front/extended layers
      - A SWAP that itself involves highly-idle qubits earns a relief discount,
        since executing it directly addresses the starvation queue.
    """
    import math

    # Lazy-initialize starvation counters (logical qubit -> idle step count)
    if not hasattr(self, 'idle_time'):
        self.idle_time = {}

    W = 1.0
    front_layer_size  = max(len(self.front_layer), 1)
    extended_layer_size = len(self.extended_layer)

    q1_swap, q2_swap = swap_gate

    # --- Front layer: urgency-weighted distance ---
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]

        # Starvation urgency: idle qubits amplify how badly this gate needs routing
        t1 = self.idle_time.get(q1, 0)
        t2 = self.idle_time.get(q2, 0)
        urgency = 1.0 + math.log1p(t1 + t2)

        f_cost += urgency * self.distance_matrix[Q1][Q2]

    # --- Extended layer: lookahead with positional decay and urgency ---
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1

        t1 = self.idle_time.get(q1, 0)
        t2 = self.idle_time.get(q2, 0)
        urgency = 1.0 + math.log1p(t1 + t2)

        e_cost += urgency * self.distance_matrix[Q1][Q2] / layer_factor

    # --- Relief factor: discount SWAPs that serve starved qubits ---
    # A SWAP on long-idle qubits is "green-lighting" a starvation queue,
    # so we reduce its cost proportionally.
    t_swap = self.idle_time.get(q1_swap, 0) + self.idle_time.get(q2_swap, 0)
    relief = 1.0 / (1.0 + math.log1p(t_swap))

    H = relief * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H