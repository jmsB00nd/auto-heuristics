# Strategy: **Qubit Routing Frustration Index**
# Intuition: Traditional cost functions are *gate-centric* — they sum distances per gate independently. Instead, we adopt a *qubit-centric* view: a logical qubit is "frustrated" when multiple pending gates pull it toward conflicting physical destinations simultaneously. We penalize the weighted **variance** of distance-demands across all pending gates per qubit — a qubit torn between a nearby gate (d=1) and a distant one (d=4) is superlinearly harder to route than two separate qubits each with one clear destination.

def qlosure_poly_heuristic(self, swap_gate):
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # Build per-logical-qubit demand lists: (distance_to_gate_partner, weight)
    qubit_demands = {}

    # Front layer: high urgency — weight = criticality
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        d = self.distance_matrix[Q1][Q2]
        crit = float(self.dag_dependencies_count[g] + 1)
        qubit_demands.setdefault(q1, []).append((d, crit))
        qubit_demands.setdefault(q2, []).append((d, crit))

    # Extended layer: discounted by lookahead depth
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        d = self.distance_matrix[Q1][Q2]
        crit = float(self.dag_dependencies_count[g] + 1)
        depth = float(self.extended_layer_index.get(g, 0) + 1)
        w = crit / depth
        qubit_demands.setdefault(q1, []).append((d, w))
        qubit_demands.setdefault(q2, []).append((d, w))

    if not qubit_demands:
        return 0.0

    total_score = 0.0
    for demands in qubit_demands.values():
        total_w = sum(w for _, w in demands)
        if total_w == 0.0:
            continue
        # Weighted mean distance: base routing cost for this qubit
        mean_d = sum(d * w for d, w in demands) / total_w
        # Weighted variance: penalizes conflicting routing directions (frustration)
        variance = sum(w * (d - mean_d) ** 2 for d, w in demands) / total_w
        total_score += mean_d + variance

    # Normalize by number of active qubits (not gates)
    H = max_decay * total_score / len(qubit_demands)
    return H