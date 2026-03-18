def qlosure_poly_heuristic(self, swap_gate):
    gamma = 0.5   # geometric decay; Σγ^k = 1/(1-γ) = 2.0 (provably finite)
    W = 0.5       # balance front vs extended (halved to compensate for γ-series scale)

    front_layer_size = max(len(self.front_layer), 1)
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Front layer: dependency-weighted hardware distance ---
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]
    f_distance /= front_layer_size

    # --- Multi-Scale Geometric Telescope over extended layer ---
    # Bucket k: gates where 2^(k-1) < layer_factor <= 2^k
    #   k = (layer_factor - 1).bit_length()  →  k=0 for layer_factor=1,
    #                                             k=1 for layer_factor=2,
    #                                             k=2 for layer_factor=3,4,
    #                                             k=3 for layer_factor=5..8, ...
    level_sums   = {}  # k -> sum of gate costs in bucket k
    level_counts = {}  # k -> |L[k]|

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1   # 1-based BFS depth
        deps = self.dag_dependencies_count[g]
        gate_cost = (deps + 1) * self.distance_matrix[Q1][Q2]

        k = max(0, (layer_factor - 1).bit_length())   # O(1), no log call needed

        if k not in level_sums:
            level_sums[k]   = 0.0
            level_counts[k] = 0
        level_sums[k]   += gate_cost
        level_counts[k] += 1

    # Σ_k  γ^k * (1/|L[k]|) * Σ_{g∈L[k]} (deps+1)*dist
    # Per-bucket normalisation prevents large buckets from dominating;
    # γ^k decay ensures bounded total weight and no runaway scaling.
    e_distance = 0.0
    for k, s in level_sums.items():
        e_distance += (gamma ** k) * (s / level_counts[k])   # level_counts[k] >= 1

    H = max_decay * (f_distance + W * e_distance)
    return H