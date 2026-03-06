# Idea: Qubit Interaction Graph Alignment Cost (QIGAC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on adder_n28__42CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    # QIGAC: Build a weighted logical interaction graph L over ALL remaining gates
    # via BFS from front_layer through dag2q, accumulating edge weights per qubit pair.
    # Cost = weighted sum of physical distances (global graph embedding distortion).

    # --- Step 1: Traverse all remaining gates and build interaction weight map ---
    interaction_weights = {}   # (min_lq, max_lq) -> total gate count

    visited = set(self.front_layer)
    queue  = list(self.front_layer)

    while queue:
        gate = queue.pop(0)
        q1, q2 = self.access2q[gate]
        key = (min(q1, q2), max(q1, q2))
        interaction_weights[key] = interaction_weights.get(key, 0) + 1

        for succ in self.dag2q.get(gate, set()):
            if succ not in visited:
                visited.add(succ)
                queue.append(succ)

    if not interaction_weights:
        return 0.0

    # --- Step 2: Compute global weighted distortion between L and physical topology ---
    total_cost   = 0.0
    total_weight = 0.0

    for (q1, q2), weight in interaction_weights.items():
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:          # unmapped qubits — skip
            continue
        dist = self.distance_matrix[Q1][Q2]
        total_cost   += weight * dist
        total_weight += weight

    if total_weight == 0.0:
        return 0.0

    # --- Step 3: Penalise hot qubits involved in this SWAP ---
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # Normalise by total interaction weight → average weighted distortion per gate
    return max_decay * total_cost / total_weight