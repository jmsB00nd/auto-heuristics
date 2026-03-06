def qlosure_poly_heuristic(self, swap_gate):
    import math

    W = 1
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # --- Step 1: Build routing pressure for every physical qubit ---
    # pressure[Q] = sum of (deps+1)*dist over all pending gates that touch Q
    pressure = {}
    for g in list(self.front_layer) + list(self.extended_layer):
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps   = self.dag_dependencies_count[g]
        burden = (deps + 1) * self.distance_matrix[Q1][Q2]
        pressure[Q1] = pressure.get(Q1, 0.0) + burden
        pressure[Q2] = pressure.get(Q2, 0.0) + burden

    # Normalise to [0,1] so the multiplier stays in a stable range
    max_p = max(pressure.values()) if pressure else 1.0
    if max_p == 0.0:
        max_p = 1.0
    norm_p = {q: v / max_p for q, v in pressure.items()}

    # --- Step 2: RGLPC — distance scaled by geometric-mean pressure ---
    # geo-mean factor ∈ [1, 2]; high pressure ↔ higher cost multiplier

    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps    = self.dag_dependencies_count[g]
        p1      = norm_p.get(Q1, 0.0) + 1.0   # +1 → minimum factor of 1
        p2      = norm_p.get(Q2, 0.0) + 1.0
        geo_p   = math.sqrt(p1 * p2)
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2] * geo_p

    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps    = self.dag_dependencies_count[g]
        p1      = norm_p.get(Q1, 0.0) + 1.0
        p2      = norm_p.get(Q2, 0.0) + 1.0
        geo_p   = math.sqrt(p1 * p2)
        e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] * geo_p / layer_factor

    H = max_decay * (
        f_distance / front_layer_size
        + W * ((e_distance / extended_layer_size) if extended_layer_size else 0.0)
    )

    return H