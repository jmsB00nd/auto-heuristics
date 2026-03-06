# Idea: Stochastic DAG Activation Probability Cost (SDAPC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on qft_n29__222CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    front = self.front_layer
    extended = self.extended_layer

    if not front:
        return 0.0

    # ── Horizon K and sigmoid temperature ────────────────────────────────────
    # K adapts to circuit complexity: more outstanding gates → wider horizon
    n_ext = len(extended)
    K = max(4, n_ext // 2 + 3)
    TAU = 1.5  # sigmoid sharpness; larger = softer probability boundary

    all_gates = set(front) | set(extended)

    # ── DP: compute T[g] = minimum routing steps before g can execute ────────
    #
    #  Front-layer gates have no unresolved 2q predecessors in scope:
    #    T[g] = max(0, dist(g) - 1)        (just need to route the two qubits)
    #
    #  Extended-layer gates depend on predecessors finishing first:
    #    T[g] = max_{p in preds ∩ scope}(T[p] + 1)  +  max(0, dist(g) - 1)
    #          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^^
    #          wait for blocking predecessor to execute    then route own qubits
    #
    #  The +1 accounts for the execution step of the predecessor itself.
    #  Using only predecessors within the visible scope prevents unbounded
    #  recursion while still propagating the critical path faithfully.

    min_steps = {}  # gate_id -> T[g]

    # Front layer: immediately runnable, cost = routing distance only
    for g in front:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            min_steps[g] = 0
        else:
            min_steps[g] = max(0, self.distance_matrix[Q1][Q2] - 1)

    # Extended layer: sort by lookahead depth so predecessors are always
    # resolved before successors (extended_layer_index gives topological order)
    ext_ordered = sorted(extended, key=lambda g: self.extended_layer_index.get(g, 0))

    for g in ext_ordered:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            routing_cost = 0
        else:
            routing_cost = max(0, self.distance_matrix[Q1][Q2] - 1)

        # Critical-path constraint: must wait for the slowest predecessor
        preds_in_scope = self.dag_predecessors2q.get(g, set()) & all_gates
        if preds_in_scope:
            # +1: predecessor must fully execute before g can start routing
            pred_barrier = max(min_steps.get(p, 0) + 1 for p in preds_in_scope)
        else:
            pred_barrier = 0

        min_steps[g] = pred_barrier + routing_cost

    # ── Activation probabilities P(g, K) via logistic model ─────────────────
    #
    #  slack(g) = K - T[g]
    #    > 0 : feasible with room to spare  → P → 1
    #    = 0 : exactly on the horizon       → P = 0.5
    #    < 0 : requires more steps than K   → P → 0
    #
    #  P(g, K) = sigmoid(slack / TAU) = 1 / (1 + exp(-slack / TAU))
    #
    #  Clamp exponent to [-50, 50] to prevent float overflow.

    def activation_prob(g):
        t = min_steps.get(g, 0)
        slack = K - t
        x = max(-50.0, min(50.0, slack / TAU))
        return 1.0 / (1.0 + math.exp(-x))

    # ── Probability-weighted expected distance ───────────────────────────────
    #
    #  Each gate contributes:  P(g) * criticality(g) * dist(g)
    #  Normalised by:          sum of P(g) * criticality(g)
    #
    #  Interpretation: the cost is the *expected routing distance* under the
    #  stochastic activation model, giving higher weight to critical gates
    #  (many dependents) that are likely to activate soon.

    weighted_dist = 0.0
    weight_total = 0.0

    for g in all_gates:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue

        dist = self.distance_matrix[Q1][Q2]
        prob = activation_prob(g)

        # Criticality: gates with more downstream dependents are more urgent
        criticality = self.dag_dependencies_count[g] + 1

        weighted_dist += prob * criticality * dist
        weight_total += prob * criticality

    if weight_total == 0.0:
        return 0.0

    # Expected distance under the activation-probability distribution
    expected_dist = weighted_dist / weight_total

    # ── Qubit health penalty ─────────────────────────────────────────────────
    # Penalise SWAPs on already-hot qubits to spread circuit load
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    return max_decay * expected_dist