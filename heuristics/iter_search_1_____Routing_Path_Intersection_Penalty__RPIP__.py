# Strategy: ** `Routing Path Intersection Penalty (RPIP)`
# Intuition: ** A SWAP that forces multiple high-criticality gate pairs to route through the same intermediate physical qubits creates compounding congestion — those qubits become sequencing bottlenecks that delay all dependent paths. By measuring *superlinear* load on each intermediate qubit (squared accumulation), we penalize configurations that create routing chokepoints, since two demands competing for the same intermediate node is worse than 2× one demand (convex congestion cost, analogous to network flow theory).

def qlosure_poly_heuristic(self, swap_gate):
    dm      = self.distance_matrix
    mapping = self.temp_mapping_dict   # post-swap mapping
    n       = self.num_qubits

    # ---------------------------------------------------------------
    # Build path_load[r]: total criticality-weighted demand
    # routed *through* physical qubit r across all visible gates.
    #
    # A physical qubit r is on a shortest path p1→p2 iff:
    #   dist(p1, r) + dist(r, p2) == dist(p1, p2)
    # This uses only the precomputed distance_matrix — no BFS needed.
    # ---------------------------------------------------------------
    path_load = [0.0] * n

    # --- Front layer: full criticality weight ---
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        p1, p2 = mapping[q1], mapping[q2]
        d = dm[p1][p2]
        if d == 0:
            continue  # already adjacent — no intermediate load

        crit = self.dag_dependencies_count[g] + 1

        for r in range(n):
            if dm[p1][r] + dm[r][p2] == d:
                path_load[r] += crit

    # --- Extended layer: discounted by lookahead depth ---
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        p1, p2 = mapping[q1], mapping[q2]
        d = dm[p1][p2]
        if d == 0:
            continue

        depth = self.extended_layer_index.get(g, 1)
        crit  = self.dag_dependencies_count[g] + 1
        w     = crit / (depth + 1)          # harmonic depth discount

        for r in range(n):
            if dm[p1][r] + dm[r][p2] == d:
                path_load[r] += w

    # ---------------------------------------------------------------
    # Superlinear (squared) congestion aggregation.
    # Squaring means: one qubit bearing load 4 costs 16,
    # while four qubits each bearing load 1 cost only 4.
    # This nonlinearly punishes routing chokepoints.
    # ---------------------------------------------------------------
    total_gates = len(self.front_layer) + len(self.extended_layer)
    norm        = max(total_gates, 1)

    congestion = sum(load * load for load in path_load) / norm

    # Thermal penalty: prefer swaps on "cool" qubits
    heat = max(self.decay_parameter[swap_gate[0]],
               self.decay_parameter[swap_gate[1]])

    return heat * congestion