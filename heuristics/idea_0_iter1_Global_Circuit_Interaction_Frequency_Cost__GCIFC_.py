# Idea: Global Circuit Interaction Frequency Cost (GCIFC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on multiplier_n75__1308CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    from collections import deque

    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Build global future interaction frequency beyond F∪E ---
    # BFS through DAG successors of extended_layer, skipping F∪E gates
    fe_gates = set(self.front_layer) | set(self.extended_layer)
    visited = set(fe_gates)
    queue = deque()

    for g in self.extended_layer:
        for succ in self.dag2q.get(g, set()):
            if succ not in visited:
                queue.append(succ)
                visited.add(succ)

    # Count how many times each logical qubit pair interacts in the tail circuit
    future_pair_freq = {}
    while queue:
        g = queue.popleft()
        q1, q2 = self.access2q[g]
        pair = (min(q1, q2), max(q1, q2))
        future_pair_freq[pair] = future_pair_freq.get(pair, 0) + 1
        for succ in self.dag2q.get(g, set()):
            if succ not in visited:
                queue.append(succ)
                visited.add(succ)

    # --- Front layer: distance weighted by downstream interaction frequency ---
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        pair = (min(q1, q2), max(q1, q2))
        # Pairs with many future interactions get high weight → keep them close
        freq_weight = future_pair_freq.get(pair, 0) + 1
        f_distance += freq_weight * self.distance_matrix[Q1][Q2]

    # --- Extended layer: frequency-weighted + depth decay ---
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        pair = (min(q1, q2), max(q1, q2))
        freq_weight = future_pair_freq.get(pair, 0) + 1
        layer_depth = self.extended_layer_index.get(g, 0) + 1
        # Frequency amplifies urgency; depth attenuates it for distant lookahead
        e_distance += freq_weight * self.distance_matrix[Q1][Q2] / layer_depth

    H = max_decay * (
        f_distance / front_layer_size +
        (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    return H