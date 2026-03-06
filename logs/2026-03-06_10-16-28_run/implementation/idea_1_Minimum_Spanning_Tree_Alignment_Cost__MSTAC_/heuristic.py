def qlosure_poly_heuristic(self, swap_gate):
    # --- Build weighted interaction graph from remaining circuit ---
    # Edge weight = total "urgency-adjusted" interactions between logical qubit pairs
    interaction_weights = {}

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        key = (min(q1, q2), max(q1, q2))
        w = self.dag_dependencies_count[g] + 1
        interaction_weights[key] = interaction_weights.get(key, 0) + w

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        key = (min(q1, q2), max(q1, q2))
        layer_idx = self.extended_layer_index.get(g, 0) + 1
        w = (self.dag_dependencies_count[g] + 1) / layer_idx
        interaction_weights[key] = interaction_weights.get(key, 0) + w

    if not interaction_weights:
        return 0.0

    # --- Collect all logical qubits in the interaction graph ---
    qubits = set()
    for q1, q2 in interaction_weights:
        qubits.add(q1)
        qubits.add(q2)

    # --- Maximum Spanning Tree via Kruskal's ---
    # Heavier edges = more critical interactions → kept in the skeleton
    edges = sorted(interaction_weights.items(), key=lambda x: x[1], reverse=True)

    parent = {q: q for q in qubits}
    rank   = {q: 0 for q in qubits}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True

    mst_edges = []
    for (q1, q2), w in edges:
        if union(q1, q2):
            mst_edges.append(((q1, q2), w))
        if len(mst_edges) == len(qubits) - 1:
            break

    if not mst_edges:
        return 0.0

    # --- MST Alignment Cost ---
    # Measures how well the interaction skeleton maps onto physical topology.
    # Each MST edge contributes (normalized_weight × physical_distance).
    # A mapping that embeds the skeleton with low stretch → low cost.
    total_interaction = sum(w for _, w in mst_edges)

    mst_alignment_cost = 0.0
    for (q1, q2), w in mst_edges:
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        phys_dist = self.distance_matrix[Q1][Q2]
        mst_alignment_cost += (w / total_interaction) * phys_dist

    # Apply decay penalty for overused physical qubits
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    return max_decay * mst_alignment_cost