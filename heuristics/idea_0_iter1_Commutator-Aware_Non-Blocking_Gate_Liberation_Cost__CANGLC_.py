# Idea: Commutator-Aware Non-Blocking Gate Liberation Cost (CANGLC)
# Stats: {"mean_swaps": 673.9545454545455, "mean_depth": 1110.8181818181818, "mean_runtime": 24.580664710565046, "total_circuits": 22, "successful_runs": 22, "failed_runs": 0, "error": null, "first_failure_error": null, "first_failure_traceback": null}

def qlosure_poly_heuristic(self, swap_gate):
    # ------------------------------------------------------------------ #
    # Commutativity check                                                  #
    #   Two 2-qubit gates COMMUTE (are phantom dependencies) when they    #
    #   act on completely disjoint qubit sets.  Gates sharing at least    #
    #   one qubit are treated as NON-COMMUTING — a genuine ordering       #
    #   constraint that cannot be reordered away.                         #
    # ------------------------------------------------------------------ #
    def shares_qubit(g1, g2):
        qa, qb = self.access2q[g1]
        qc, qd = self.access2q[g2]
        return bool({qa, qb} & {qc, qd})

    # ------------------------------------------------------------------ #
    # Non-commuting predecessor chain depth  (nc_depth)                   #
    #   Recursively walk BACKWARD through DAG predecessors, following     #
    #   only edges where the predecessor shares a qubit with the current  #
    #   gate (non-commuting pair).  Returns the length of the longest     #
    #   such chain — the algebraic mandatory-ordering depth.              #
    #                                                                      #
    #   depth_limit caps recursion at MAX_CHAIN_DEPTH to remain O(1) per #
    #   gate in practice; 5 levels captures all practically relevant      #
    #   ordering constraints while avoiding DAG-wide traversal.           #
    # ------------------------------------------------------------------ #
    MAX_CHAIN_DEPTH = 5
    _cache = {}

    def nc_depth(gate_id, remaining):
        if remaining == 0:
            return 1
        key = (gate_id, remaining)
        if key in _cache:
            return _cache[key]

        preds = self.dag_predecessors2q.get(gate_id, set())
        # Keep only predecessors that are non-commuting with gate_id
        nc_preds = [p for p in preds if shares_qubit(gate_id, p)]

        if not nc_preds:
            _cache[key] = 1
            return 1

        depth = 1 + max(nc_depth(p, remaining - 1) for p in nc_preds)
        _cache[key] = depth
        return depth

    # ------------------------------------------------------------------ #
    # Decay and sizes                                                      #
    # ------------------------------------------------------------------ #
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )
    fl_size = max(len(self.front_layer), 1)
    el_size = max(len(self.extended_layer), 1)

    # ------------------------------------------------------------------ #
    # Front-layer: algebraic-depth-weighted distance                      #
    #   nc_depth(g) replaces the raw dag_dependencies_count.  A gate     #
    #   whose predecessors are all commuting has nc_depth == 1 (it can   #
    #   be freely reordered); a gate deep in a non-commuting chain has    #
    #   high nc_depth and therefore high urgency.                         #
    # ------------------------------------------------------------------ #
    f_score = 0.0
    liberation_pressure = 0.0   # nc_depth of gates that become executable

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        nd = nc_depth(g, MAX_CHAIN_DEPTH)

        f_score += nd * dist

        # Liberation: if this SWAP places g at distance 1 (ready to fire)
        # its non-commuting chain pressure is about to collapse — record
        # the relief it will deliver to successors.
        if dist == 1:
            liberation_pressure += nd

    # ------------------------------------------------------------------ #
    # Extended-layer: discounted algebraic-depth-weighted distance        #
    #   Layer position discounts future gates; nc_depth still reflects    #
    #   their true sequencing cost.                                        #
    # ------------------------------------------------------------------ #
    e_score = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist = self.distance_matrix[Q1][Q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        nd = nc_depth(g, MAX_CHAIN_DEPTH)

        e_score += nd * dist / layer_factor

    # ------------------------------------------------------------------ #
    # Liberation factor  (denominator term)                               #
    #   SWAPs that immediately free high-depth non-commuting gates earn   #
    #   a lower cost.  Normalise by front-layer size so the factor is     #
    #   dimensionless and scale-invariant.                                 #
    # ------------------------------------------------------------------ #
    liberation_factor = liberation_pressure / fl_size

    # ------------------------------------------------------------------ #
    # CANGLC cost                                                          #
    #                                                                      #
    #   cost = decay  ×  (f_score/|F|  +  0.5 × e_score/|E|)            #
    #                  ÷  (1 + liberation_factor)                         #
    #                                                                      #
    #   Numerator  : urgency of remaining non-commuting ordering work.    #
    #   Denominator: reward for liberating constrained non-commuting      #
    #                chains now, preventing cascading SWAPs later.        #
    # ------------------------------------------------------------------ #
    W_ext = 0.5
    numerator = max_decay * (f_score / fl_size + W_ext * e_score / el_size)
    cost = numerator / (1.0 + liberation_factor)

    return cost