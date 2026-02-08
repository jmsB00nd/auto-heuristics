# Strategy: "Harmonic Criticality with Congestion Penalty"
# Intuition: Instead of linearly weighting by dependency count, use a harmonic mean of distances weighted by criticality squared, and add a congestion penalty that detects when multiple front-layer gates compete for the same physical qubits — penalizing swaps that create bottlenecks rather than just minimizing aggregate distance.
# Stats: {'mean_swaps': 588.0454545454545, 'mean_depth': 948.4090909090909, 'mean_runtime': 3.619590629230846, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])

    # Front layer: criticality-squared weighting with harmonic distance
    f_cost = 0
    phys_usage = {}  # track how many front gates need each physical qubit
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        deps = self.dag_dependencies_count[g]

        # Square criticality to heavily prioritize high-dependency gates
        weight = (deps + 1) ** 2
        f_cost += weight * dist

        # Track physical qubit demand for congestion
        phys_usage[Q1] = phys_usage.get(Q1, 0) + 1
        phys_usage[Q2] = phys_usage.get(Q2, 0) + 1

    # Congestion penalty: if the swap targets are highly contested physical qubits
    congestion = 0
    for pq in (swap_gate[0], swap_gate[1]):
        if pq in phys_usage:
            congestion += phys_usage[pq] - 1  # extra demand beyond 1

    # Extended layer: inverse-square layer depth decay with criticality
    e_cost = 0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_depth = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]

        # Inverse square decay by depth — near-future gates matter much more
        e_cost += (deps + 1) * self.distance_matrix[Q1][Q2] / (layer_depth ** 2)

    W = 0.5  # reduced lookahead weight
    C = 0.3  # congestion penalty weight

    f_norm = f_cost / front_layer_size if front_layer_size else 0
    e_norm = e_cost / extended_layer_size if extended_layer_size else 0

    H = max_decay * (f_norm + W * e_norm + C * congestion)

    return H