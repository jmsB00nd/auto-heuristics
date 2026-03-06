# Idea: \_NAME: Edge Congestion Anticipation Cost (ECAC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on adder_n28__42CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    p0, p1 = swap_gate[0], swap_gate[1]
    swap_edge = (min(p0, p1), max(p0, p1))

    front_layer_size    = max(len(self.front_layer), 1)
    extended_layer_size = len(self.extended_layer)

    max_decay = max(self.decay_parameter[p0], self.decay_parameter[p1])

    # ------------------------------------------------------------------ #
    # Step 1: Build edge-load map                                          #
    #                                                                      #
    # load(e) = |{ g ∈ F∪E : e lies on some shortest path for g }|        #
    #                                                                      #
    # Membership test (no path walking):                                   #
    #   edge (a,b) lies on a shortest path from Q1→Q2  iff               #
    #       dist(Q1,a) + 1 + dist(b,Q2) == d                              #
    #    OR dist(Q1,b) + 1 + dist(a,Q2) == d                              #
    # This exploits the triangle-equality characterisation of all          #
    # shortest-path edges without enumerating explicit routes.             #
    # ------------------------------------------------------------------ #
    edge_load = {}
    dm = self.distance_matrix            # local alias for speed
    edges = list(self.backend_connections)  # evaluated once

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0 or Q1 == Q2:
            continue
        d = dm[Q1][Q2]
        for (a, b) in edges:
            if (dm[Q1][a] + 1 + dm[b][Q2] == d or
                    dm[Q1][b] + 1 + dm[a][Q2] == d):
                key = (a, b)
                edge_load[key] = edge_load.get(key, 0) + 1

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0 or Q1 == Q2:
            continue
        d = dm[Q1][Q2]
        for (a, b) in edges:
            if (dm[Q1][a] + 1 + dm[b][Q2] == d or
                    dm[Q1][b] + 1 + dm[a][Q2] == d):
                key = (a, b)
                edge_load[key] = edge_load.get(key, 0) + 1

    # Raw gate-count load on the SWAP edge; normalised ∈ [0, 1]
    total_gates  = front_layer_size + extended_layer_size
    swap_load    = edge_load.get(swap_edge, 0)
    norm_load    = swap_load / total_gates          # ∈ [0, 1]

    # ------------------------------------------------------------------ #
    # Step 2: H_baseline — standard weighted-distance term                #
    #                                                                      #
    # Mirrors the reference heuristic but uses only (deps+1) weighting    #
    # without the extended-layer depth discount (kept in e_distance).      #
    # ------------------------------------------------------------------ #
    W = 1.0

    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * dm[Q1][Q2]

    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        e_distance += (deps + 1) * dm[Q1][Q2] / layer_factor

    H_baseline = (
        f_distance / front_layer_size
        + W * (e_distance / extended_layer_size if extended_layer_size else 0.0)
    )

    # ------------------------------------------------------------------ #
    # Step 3: ECAC formula                                                 #
    #                                                                      #
    #   H_ECAC = max_decay × H_baseline + λ × load(SWAP_edge)            #
    #                                                                      #
    # λ is scaled by H_baseline so the congestion penalty is always        #
    # proportional to the current routing difficulty — avoiding the term   #
    # dominating on easy circuits or vanishing on hard ones.               #
    # ------------------------------------------------------------------ #
    lam = 0.5
    H = max_decay * H_baseline + lam * norm_load * (H_baseline + 1.0)

    return H