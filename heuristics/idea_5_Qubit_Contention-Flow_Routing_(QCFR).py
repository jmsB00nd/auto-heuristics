# Strategy: Qubit Contention-Flow Routing (QCFR)
# Intuition: Logical qubits that appear in many pending gates are routing bottlenecks — resolving a gate involving a high-contention qubit cascades benefits by freeing that qubit for subsequent operations. By computing per-qubit "contention pressure" from all visible gates and using it to amplify the distance-reduction signal, the router prioritizes swaps that relieve multi-gate bottleneck qubits rather than those helping isolated gates.

## What makes this DISTINCT from all tested approaches:

| Prior Approach | Key Idea | QCFR Difference |
|---|---|---|
| AGLR-IP | Per-gate gain/loss ratios + interference | QCFR weights by per-*qubit* contention across gates, not per-gate deltas |
| SPAP | Swap alignment on hardware shortest paths | QCFR is about logical qubit reuse frequency, not topological alignment |
| CADE | Adaptive distance exponent per criticality | QCFR adds an orthogonal dimension: qubit contention pressure, not exponent modulation |
| DWDP | Per-qubit demand centroids (where to be) | QCFR measures *how contested* a qubit is (how many gates need it), not *where* it should go |

##
# Stats: {'mean_swaps': 703.1363636363636, 'mean_depth': 1014.0909090909091, 'mean_runtime': 9.625368692658164, 'total_circuits': 22, 'successful_runs': 22, 'failed_runs': 0, 'error': None}

def qlosure_poly_heuristic(self, swap_gate):
    import math

    # --- Hyperparameters ---
    CRIT_PWR = 0.68          # Sub-linear criticality scaling
    CONTENTION_PWR = 0.55    # Sub-linear contention scaling (prevents runaway)
    DIST_EXP = 0.90          # Distance exponent (slightly sub-linear)
    GAUSS_2SIG2 = 9.0        # Gaussian depth decay 2*sigma^2
    LOOKAHEAD_W = 0.65       # Extended layer weight
    DELTA_BONUS_W = 0.45     # Weight for contention-amplified delta bonus

    p1, p2 = swap_gate
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # ---------------------------------------------------------------
    # Phase 1: Build per-logical-qubit contention pressure
    # Count how many visible gates each logical qubit participates in,
    # weighted by criticality and urgency (depth decay).
    # ---------------------------------------------------------------
    contention = {}  # logical_qubit -> accumulated pressure

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        deps = self.dag_dependencies_count[g]
        w = (deps + 1.0) ** CRIT_PWR
        contention[q1] = contention.get(q1, 0.0) + w
        contention[q2] = contention.get(q2, 0.0) + w

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        deps = self.dag_dependencies_count[g]
        depth = self.extended_layer_index.get(g, 0)
        decay = math.exp(-(depth ** 2) / GAUSS_2SIG2)
        w = (deps + 1.0) ** CRIT_PWR * decay * LOOKAHEAD_W
        contention[q1] = contention.get(q1, 0.0) + w
        contention[q2] = contention.get(q2, 0.0) + w

    # ---------------------------------------------------------------
    # Phase 2: Front Layer — contention-weighted distance cost
    # plus contention-amplified delta bonus
    # ---------------------------------------------------------------
    f_cost = 0.0
    f_delta = 0.0
    f_count = 0

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist_after = self.distance_matrix[Q1][Q2]

        deps = self.dag_dependencies_count[g]
        crit_w = (deps + 1.0) ** CRIT_PWR

        # Per-gate contention: geometric mean of both qubits' contention
        c1 = contention.get(q1, 1.0)
        c2 = contention.get(q2, 1.0)
        gate_contention = ((c1 * c2) ** 0.5) ** CONTENTION_PWR

        # Base cost: contention-weighted distance
        f_cost += crit_w * gate_contention * (dist_after ** DIST_EXP)

        # Delta term: contention-amplified distance reduction
        P1_before = self.mapping_dict[q1]
        P2_before = self.mapping_dict[q2]
        dist_before = self.distance_matrix[P1_before][P2_before]
        delta = dist_before - dist_after  # positive = improvement

        # Only reward improvements, scaled by contention
        if delta > 0:
            f_delta += delta * gate_contention * crit_w

        f_count += 1

    # ---------------------------------------------------------------
    # Phase 3: Extended Layer — same logic with Gaussian decay
    # ---------------------------------------------------------------
    e_cost = 0.0
    e_delta = 0.0
    e_count = 0

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        dist_after = self.distance_matrix[Q1][Q2]

        deps = self.dag_dependencies_count[g]
        depth = self.extended_layer_index.get(g, 0)
        decay = math.exp(-(depth ** 2) / GAUSS_2SIG2)

        crit_w = (deps + 1.0) ** CRIT_PWR

        c1 = contention.get(q1, 1.0)
        c2 = contention.get(q2, 1.0)
        gate_contention = ((c1 * c2) ** 0.5) ** CONTENTION_PWR

        e_cost += crit_w * gate_contention * (dist_after ** DIST_EXP) * decay

        P1_before = self.mapping_dict[q1]
        P2_before = self.mapping_dict[q2]
        dist_before = self.distance_matrix[P1_before][P2_before]
        delta = dist_before - dist_after

        if delta > 0:
            e_delta += delta * gate_contention * crit_w * decay

        e_count += 1

    # ---------------------------------------------------------------
    # Phase 4: Aggregation
    # ---------------------------------------------------------------
    h_f = (f_cost / f_count) if f_count > 0 else 0.0
    h_e = (e_cost / e_count) if e_count > 0 else 0.0
    d_f = (f_delta / f_count) if f_count > 0 else 0.0
    d_e = (e_delta / e_count) if e_count > 0 else 0.0

    # Cost (minimize) minus delta bonus (maximize -> subtract)
    cost_term = h_f + LOOKAHEAD_W * h_e
    delta_term = d_f + LOOKAHEAD_W * d_e

    H = max_decay * (cost_term - DELTA_BONUS_W * delta_term)
    return float(H)