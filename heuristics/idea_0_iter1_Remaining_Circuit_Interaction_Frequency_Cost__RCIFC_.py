# Idea: Remaining Circuit Interaction Frequency Cost (RCIFC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on multiplier_n75__1308CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    # --- Step 1: BFS over entire remaining circuit to build pair interaction frequencies ---
    pair_frequency = {}
    visited = set(self.front_layer)
    queue = list(self.front_layer)

    while queue:
        g = queue.pop(0)
        q1, q2 = self.access2q[g]
        pair = (min(q1, q2), max(q1, q2))
        pair_frequency[pair] = pair_frequency.get(pair, 0) + 1

        for successor in self.dag2q.get(g, set()):
            if successor not in visited:
                visited.add(successor)
                queue.append(successor)

    # --- Step 2: Front layer cost — weight distance by global remaining frequency ---
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        pair = (min(q1, q2), max(q1, q2))
        freq = pair_frequency.get(pair, 1)
        f_distance += freq * self.distance_matrix[Q1][Q2]

    # --- Step 3: Extended layer cost — frequency weight + lookahead depth decay ---
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        pair = (min(q1, q2), max(q1, q2))
        freq = pair_frequency.get(pair, 1)
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        e_distance += freq * self.distance_matrix[Q1][Q2] / layer_factor

    H = max_decay * (
        f_distance / front_layer_size +
        (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    return H