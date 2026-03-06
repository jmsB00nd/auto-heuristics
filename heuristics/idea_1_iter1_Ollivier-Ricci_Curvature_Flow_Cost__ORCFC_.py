# Idea: Ollivier-Ricci Curvature Flow Cost (ORCFC)
# Stats: {"mean_swaps": 962.1818181818181, "mean_depth": 1161.6818181818182, "mean_runtime": 2.3487371748143975, "total_circuits": 22, "successful_runs": 22, "failed_runs": 0, "error": null, "first_failure_error": null, "first_failure_traceback": null}

def qlosure_poly_heuristic(self, swap_gate):
    # ── Build curvature-adjusted all-pairs distances (cached after first call) ──
    if not hasattr(self, '_orcfc_distances'):
        n   = self.num_qubits
        INF = float('inf')

        # Forman-Ricci curvature proxy for each hardware edge (u,v):
        #   κ(u,v) = 2 − deg(u) − deg(v) + |N(u) ∩ N(v)|
        # Strongly negative  → bridge / bottleneck  (penalise)
        # Near-zero/positive → clique-like region   (prefer)
        curvatures = {}
        for u in range(n):
            nbrs_u = self.backend[u]
            deg_u  = len(nbrs_u)
            for v in nbrs_u:
                if (u, v) not in curvatures:
                    nbrs_v = self.backend[v]
                    deg_v  = len(nbrs_v)
                    common = len(set(nbrs_u) & set(nbrs_v))
                    kappa  = 2 - deg_u - deg_v + common
                    curvatures[(u, v)] = kappa
                    curvatures[(v, u)] = kappa

        # Normalise curvatures to [0, 1]: 1 = well-connected, 0 = bottleneck
        if curvatures:
            vals       = list(curvatures.values())
            min_k, max_k = min(vals), max(vals)
            span       = max(max_k - min_k, 1e-9)
            norm_k     = {e: (k - min_k) / span for e, k in curvatures.items()}
        else:
            norm_k = {}

        # Curvature-adjusted edge weight:
        #   w(u,v) = 1 + α·(1 − κ_norm)  ∈ [1, 1+α]
        # Low normalised curvature (bottleneck) → high traversal cost
        alpha    = 2.0
        adj_dist = [[INF] * n for _ in range(n)]
        for i in range(n):
            adj_dist[i][i] = 0.0
        for u in range(n):
            for v in self.backend[u]:
                kn = norm_k.get((u, v), 0.5)
                adj_dist[u][v] = 1.0 + alpha * (1.0 - kn)

        # Floyd-Warshall: all-pairs shortest paths under curvature-adjusted weights
        for k in range(n):
            for i in range(n):
                if adj_dist[i][k] == INF:
                    continue
                for j in range(n):
                    nd = adj_dist[i][k] + adj_dist[k][j]
                    if nd < adj_dist[i][j]:
                        adj_dist[i][j] = nd

        self._orcfc_distances = adj_dist

    cdist = self._orcfc_distances

    # ── Cost accumulation ────────────────────────────────────────────────────
    W                  = 0.5
    front_layer_size   = max(len(self.front_layer),    1)
    extended_layer_size = max(len(self.extended_layer), 1)

    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # Front layer: curvature-penalised distance, scaled by gate criticality
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue
        deps    = self.dag_dependencies_count[g]
        f_cost += (deps + 1) * cdist[Q1][Q2]

    # Extended (lookahead) layer: same metric, attenuated by lookahead depth
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 < 0 or Q2 < 0:
            continue
        depth   = self.extended_layer_index.get(g, 0) + 1
        deps    = self.dag_dependencies_count[g]
        e_cost += (deps + 1) * cdist[Q1][Q2] / depth

    H = max_decay * (
        f_cost  / front_layer_size +
        W * e_cost / extended_layer_size
    )
    return H