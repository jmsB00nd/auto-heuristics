# Idea: Minimum Target Adjacency Matching Cost (MTAMC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on swap_test_n41__23CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    front_layer_size = len(self.front_layer)
    if front_layer_size == 0:
        return 0.0

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # Phase 1: For each front-layer gate, find the hardware edge (P1, P2)
    # that minimizes d(Q1,P1) + d(Q2,P2) — the "best target" for routing.
    # This gives a lower bound on the SWAPs needed to make Q1,Q2 adjacent.
    gate_best_pair = {}  # gate_id -> frozenset{P1, P2}
    gate_base_cost = {}  # gate_id -> minimum routing displacement cost
    pair_demand = {}     # frozenset{P1, P2} -> number of gates targeting it

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]

        if Q1 < 0 or Q2 < 0:
            continue

        # Gate already executable — zero displacement cost
        if Q2 in self.backend[Q1]:
            best_cost = 0
            best_pair = frozenset({Q1, Q2})
        else:
            best_cost = float('inf')
            best_pair = None

            # Search over all hardware edges to find the cheapest landing spot
            for (P1, P2) in self.backend_connections:
                # Try both orientations: (Q1->P1, Q2->P2) and (Q1->P2, Q2->P1)
                c = min(
                    self.distance_matrix[Q1][P1] + self.distance_matrix[Q2][P2],
                    self.distance_matrix[Q1][P2] + self.distance_matrix[Q2][P1]
                )
                if c < best_cost:
                    best_cost = c
                    best_pair = frozenset({P1, P2})

        gate_best_pair[g] = best_pair
        gate_base_cost[g] = best_cost if best_cost != float('inf') else self.distance_matrix[Q1][Q2]

        if best_pair is not None:
            pair_demand[best_pair] = pair_demand.get(best_pair, 0) + 1

    if not gate_best_pair:
        return 0.0

    # Phase 2: Matching cost with interference
    # Core insight: a hardware edge is a shared resource. When k gates all
    # target the same edge, only one can proceed first — the remaining (k-1)
    # must either wait or re-route. We model this by scaling each gate's
    # base cost by the contention level on its chosen target edge.
    # This captures gate-gate interference invisible to pure sum-of-distances.
    total_cost = 0.0

    for g, best_pair in gate_best_pair.items():
        base = gate_base_cost[g]
        # demand = number of concurrent gates competing for this edge
        demand = pair_demand.get(best_pair, 1) if best_pair is not None else 1

        # Matching cost: routing displacement amplified by edge contention.
        # demand == 1  → no interference, cost is just the base displacement.
        # demand > 1   → congestion penalty: other gates block the same edge,
        #                 effectively multiplying the expected wait/re-route cost.
        total_cost += base * demand

    # Normalize across front layer and apply qubit thermal decay
    total_cost /= front_layer_size

    return max_decay * total_cost