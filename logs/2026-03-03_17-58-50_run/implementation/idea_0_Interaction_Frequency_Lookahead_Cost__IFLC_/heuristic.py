def qlosure_poly_heuristic(self, swap_gate):
    # BFS from front layer through remaining DAG to compute
    # global interaction frequency per logical qubit pair
    freq = {}
    visited = set()
    stack = list(self.front_layer)

    while stack:
        g = stack.pop()
        if g in visited:
            continue
        visited.add(g)
        q1, q2 = self.access2q[g]
        key = (min(q1, q2), max(q1, q2))
        freq[key] = freq.get(key, 0) + 1
        for succ in self.dag2q.get(g, set()):
            if succ not in visited:
                stack.append(succ)

    W = 1.0
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    f_distance = 0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        key = (min(q1, q2), max(q1, q2))
        interaction_freq = freq.get(key, 1)
        f_distance += interaction_freq * self.distance_matrix[Q1][Q2]

    e_distance = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        key = (min(q1, q2), max(q1, q2))
        interaction_freq = freq.get(key, 1)
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        e_distance += interaction_freq * self.distance_matrix[Q1][Q2] / layer_factor

    H = max_decay * (
        f_distance / front_layer_size
        + W * ((e_distance / extended_layer_size) if extended_layer_size else 0)
    )

    return H