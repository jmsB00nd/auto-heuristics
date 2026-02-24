# Idea: Optimal Transport Routing Cost (OTRC)
# Stats: {"mean_swaps": Infinity, "error": "Timeout on adder_n28__42CYC.json"}

def qlosure_poly_heuristic(self, swap_gate):
    dm = self.distance_matrix

    # ── 1. Build demand measure μ: where interaction is needed ────────────────
    demand_raw = {}

    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue
        pair = (min(Q1, Q2), max(Q1, Q2))
        # Front-layer: urgent, boosted weight
        urgency = float(self.dag_dependencies_count[g] + 1) * 2.0
        demand_raw[pair] = demand_raw.get(pair, 0.0) + urgency

    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue
        pair = (min(Q1, Q2), max(Q1, Q2))
        lookahead_depth = self.extended_layer_index.get(g, 0) + 1
        urgency = float(self.dag_dependencies_count[g] + 1) / lookahead_depth
        demand_raw[pair] = demand_raw.get(pair, 0.0) + urgency

    if not demand_raw:
        return 0.0

    total_demand = sum(demand_raw.values())
    demand_pts = list(demand_raw.keys())
    demand_weights = [demand_raw[p] / total_demand for p in demand_pts]

    # ── 2. Build supply measure ν: coupling-graph edges weighted by load ──────
    edges = list({(min(a, b), max(a, b)) for a, b in self.backend_connections})
    n_edges = len(edges)
    if n_edges == 0:
        return 0.0

    # Prefer edges on shallower (less-loaded) qubits — inverse depth weighting
    supply_raw = []
    for a, b in edges:
        da = self.qubit_depth.get(a, 0)
        db = self.qubit_depth.get(b, 0)
        supply_raw.append(1.0 / (1.0 + da + db))
    total_supply = sum(supply_raw)
    supply_weights = [w / total_supply for w in supply_raw]

    # ── 3. Ground metric: minimum SWAP work to align pair onto edge ───────────
    # d((Q1,Q2), (e1,e2)) = min alignment cost over both orientations
    def pair_dist(p, e):
        p1, p2 = p
        e1, e2 = e
        return min(dm[p1][e1] + dm[p2][e2], dm[p1][e2] + dm[p2][e1])

    # ── 4. Wasserstein-1 via greedy transport ─────────────────────────────────
    # Sort all (cost, demand_idx, supply_idx) triples ascending,
    # then greedily assign flow — this is the standard "northwest corner"
    # generalization that gives exact W1 when costs satisfy the Monge property.
    cost_entries = []
    for i, dp in enumerate(demand_pts):
        for j, ep in enumerate(edges):
            cost_entries.append((pair_dist(dp, ep), i, j))
    cost_entries.sort()

    rem_d = list(demand_weights)
    rem_s = list(supply_weights)
    w1_cost = 0.0

    for c, i, j in cost_entries:
        if rem_d[i] < 1e-12 or rem_s[j] < 1e-12:
            continue
        flow = min(rem_d[i], rem_s[j])
        w1_cost += flow * c
        rem_d[i] -= flow
        rem_s[j] -= flow

    # ── 5. Scale by thermal noise (decay) of the candidate SWAP qubits ───────
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    return max_decay * w1_cost