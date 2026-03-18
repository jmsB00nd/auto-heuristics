def qlosure_poly_heuristic(self, swap_gate):
    SWAP_COST    = 3.0    # A SWAP decomposes into 3 CNOT-equivalent gates
    LARGE_PENALTY = 1e9   # Applied when the swap yields zero or negative net improvement
    LOOKAHEAD_W  = 0.5    # Relative weight of extended-layer contribution

    front_layer_size    = len(self.front_layer)                     # guaranteed > 0 (caller ensures this)
    extended_layer_size = len(self.extended_layer)                  # may be 0

    # ------------------------------------------------------------------ #
    # 1.  Criticality-weighted distance IMPROVEMENT for the front layer   #
    #     improvement_g = dist(mapping) - dist(temp_mapping)             #
    #     A positive value means this swap brings the gate's operands     #
    #     closer together on the hardware graph.                          #
    # ------------------------------------------------------------------ #
    front_improvement = 0.0
    for g in self.front_layer:
        q1, q2    = self.access2q[g]
        dist_before = self.distance_matrix[self.mapping_dict[q1]][self.mapping_dict[q2]]
        dist_after  = self.distance_matrix[self.temp_mapping_dict[q1]][self.temp_mapping_dict[q2]]
        criticality = self.dag_dependencies_count[g] + 1   # +1 avoids zero-weight for leaf gates
        front_improvement += criticality * (dist_before - dist_after)

    # ------------------------------------------------------------------ #
    # 2.  Criticality-weighted improvement for the extended (look-ahead)  #
    #     layer, discounted by BFS depth so nearer gates dominate.        #
    # ------------------------------------------------------------------ #
    ext_improvement = 0.0
    for g in self.extended_layer:
        q1, q2    = self.access2q[g]
        dist_before = self.distance_matrix[self.mapping_dict[q1]][self.mapping_dict[q2]]
        dist_after  = self.distance_matrix[self.temp_mapping_dict[q1]][self.temp_mapping_dict[q2]]
        layer_depth = self.extended_layer_index.get(g, 0) + 1   # 1-based depth, never 0
        criticality = self.dag_dependencies_count[g] + 1
        ext_improvement += criticality * (dist_before - dist_after) / layer_depth

    # ------------------------------------------------------------------ #
    # 3.  Normalise each layer independently to prevent size bias, then   #
    #     combine into a single net-improvement scalar.                   #
    # ------------------------------------------------------------------ #
    net_improvement = (
        front_improvement / front_layer_size
        + LOOKAHEAD_W * (ext_improvement / extended_layer_size if extended_layer_size else 0.0)
    )

    # ------------------------------------------------------------------ #
    # 4.  SPRC: ratio of fixed hardware cost to net improvement.          #
    #     Swaps that improve distances more are cheaper per CNOT unit.    #
    #     Degenerate (non-improving) swaps receive a hard penalty to      #
    #     avoid undefined division and steer the search away.             #
    # ------------------------------------------------------------------ #
    if net_improvement <= 0.0:
        cost = LARGE_PENALTY
    else:
        cost = SWAP_COST / net_improvement

    # ------------------------------------------------------------------ #
    # 5.  Anti-oscillation: the decay parameter grows each time a qubit   #
    #     is swapped, making it progressively more expensive to re-swap   #
    #     the same qubit pair and breaking ping-pong cycles.              #
    # ------------------------------------------------------------------ #
    max_decay = max(self.decay_parameter[swap_gate[0]], self.decay_parameter[swap_gate[1]])
    cost *= max_decay

    # ------------------------------------------------------------------ #
    # 6.  Deterministic lexicographic tie-breaker.                        #
    #     When two swaps score identically (after decay), the one with    #
    #     the smaller (q0, q1) index pair wins — guaranteeing a unique    #
    #     minimum and preventing non-convergence from arbitrary choices.  #
    # ------------------------------------------------------------------ #
    cost += (swap_gate[0] * self.num_qubits + swap_gate[1]) * 1e-12

    return cost