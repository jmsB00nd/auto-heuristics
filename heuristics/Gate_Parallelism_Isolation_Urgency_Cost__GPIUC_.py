def qlosure_poly_heuristic(self, swap_gate):
    """
    Gate Parallelism Isolation Urgency Cost (GPIUC)
    
    Weights each front-layer gate's routing distance by its isolation urgency:
      - parallelism_degree(g) = # other front-layer gates sharing NO qubits with g
        (i.e., true parallel candidates once routed)
      - urgency(g) = 1 / (1 + parallelism_degree(g))
        -> isolated gates (degree=0) get urgency=1.0 (maximum)
        -> gates with many parallel partners get urgency->0 (they matter less per SWAP)
    
    This is fundamentally different from the baseline (dep-count weighting):
    the cost reflects *structural parallelism loss* rather than dependency depth.
    """
    front_layer_size = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Step 1: compute parallelism degree for each front-layer gate ---
    # parallelism_degree[g] = number of *other* front-layer gates with
    # completely disjoint qubit sets (true concurrency candidates)
    gate_qubits = {
        g: frozenset(self.access2q[g])
        for g in self.front_layer
    }

    parallelism_degree = {}
    for g in self.front_layer:
        deg = sum(
            1 for h in self.front_layer
            if h != g and gate_qubits[g].isdisjoint(gate_qubits[h])
        )
        parallelism_degree[g] = deg

    # --- Step 2: front-layer cost weighted by isolation urgency ---
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]

        # Isolated gates (degree=0) => urgency=1.0 (full weight)
        # Highly parallel gates => urgency -> 0 (low per-SWAP value)
        urgency = 1.0 / (1.0 + parallelism_degree[g])

        f_cost += urgency * dist

    f_cost /= front_layer_size

    # --- Step 3: extended-layer cost with geometric depth decay ---
    e_cost = 0.0
    if extended_layer_size > 0:
        for g in self.extended_layer:
            q1, q2 = self.access2q[g]
            Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
            depth = self.extended_layer_index.get(g, 0) + 1
            e_cost += self.distance_matrix[Q1][Q2] / depth
        e_cost /= extended_layer_size

    W = 0.5
    cost = max_decay * (f_cost + W * e_cost)

    return cost