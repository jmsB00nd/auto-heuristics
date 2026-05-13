def init_mapping(self):
    import math
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [i for i in range(N)]
    self.reverse_mapping_dict = [i for i in range(N)]

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        # ---- collect logical qubits and adjacency from QIG (or self.access fallback) ----
        logical_set = set()
        for gid, qs in self.access.items():
            for q in qs:
                if 0 <= q < N:
                    logical_set.add(q)

        qig = self.qubit_interaction_graph
        # Build logical neighbor sets and weighted degrees from QIG; backfill from self.access
        log_neighbors = defaultdict(set)
        log_weights = defaultdict(lambda: defaultdict(float))
        for u in logical_set:
            row = qig.get(u, {}) if hasattr(qig, "get") else qig[u]
            for v, w in row.items():
                if v == u or w <= 0:
                    continue
                log_neighbors[u].add(v)
                log_weights[u][v] = float(w)
                logical_set.add(v)
        # Fallback: ensure 2-qubit gate endpoints from self.access are present
        for gid, qs in self.access.items():
            if len(qs) == 2:
                a, b = qs[0], qs[1]
                if a == b:
                    continue
                log_neighbors[a].add(b)
                log_neighbors[b].add(a)
                if log_weights[a][b] == 0.0:
                    log_weights[a][b] = 1.0
                    log_weights[b][a] = 1.0
                logical_set.add(a); logical_set.add(b)

        logical_list = sorted(logical_set)
        idle_logicals = [q for q in range(N) if q not in logical_set]

        # ---- physical neighbor sets / degrees ----
        phys_neighbors = [set(self.backend.get(p, set())) if hasattr(self.backend, "get")
                          else set(self.backend[p]) for p in range(N)]
        phys_deg = [len(phys_neighbors[p]) for p in range(N)]
        max_phys_deg = max(phys_deg) if phys_deg else 1

        log_deg = {q: sum(log_weights[q].values()) for q in logical_list}
        max_log_deg = max(log_deg.values()) if log_deg else 1.0

        def entropy_from_weights(wmap):
            tot = sum(wmap.values())
            if tot <= 0:
                return 0.0
            h = 0.0
            for w in wmap.values():
                if w <= 0:
                    continue
                p = w / tot
                h -= p * math.log(p + 1e-12)
            return h

        def uniform_entropy(k):
            if k <= 0:
                return 0.0
            return math.log(k)

        log_entropy = {q: entropy_from_weights(log_weights[q]) for q in logical_list}
        phys_entropy = [uniform_entropy(phys_deg[p]) for p in range(N)]

        centrality = self.physical_centrality if hasattr(self, "physical_centrality") and self.physical_centrality else {}
        cmax = max(centrality.values()) if centrality else 1.0
        cmax = cmax if cmax > 0 else 1.0
        activity = self.logical_activity if hasattr(self, "logical_activity") else {}
        amax = max(activity.values()) if activity else 1.0
        amax = amax if amax > 0 else 1.0

        # ---- cost matrix (rows = logical including idle padding, cols = physical) ----
        alpha, beta, gamma, delta = 1.0, 0.5, 0.3, 0.4
        big = 1e6
        cost = np.full((N, N), big, dtype=float)

        # neighbor index maps for jaccard on logical side
        for li, L in enumerate(logical_list):
            negL = log_neighbors[L]
            dL = log_deg[L]
            hL = log_entropy[L]
            actL = float(activity.get(L, 0.0)) / amax
            for P in range(N):
                negP = phys_neighbors[P]
                # Jaccard dissimilarity on raw node sets (logical neighbor labels vs physical neighbor labels) — use sizes only as a structural proxy
                inter = min(len(negL), len(negP))  # structural overlap upper bound
                union = max(1, len(negL) + len(negP) - inter)
                jacc = inter / union  # in [0,1]
                jdiss = 1.0 - jacc
                ddiff = abs(dL - phys_deg[P]) / max(max_log_deg, max_phys_deg, 1.0)
                ediff = abs(hL - phys_entropy[P]) / (math.log(max(max_phys_deg, 2)) + 1e-9)
                centP = float(centrality.get(P, 0.0)) / cmax
                c = alpha * jdiss + beta * ddiff + gamma * ediff - delta * centP * actL
                cost[li, P] = c

        # idle logicals: prefer low-centrality leftovers, neutral structural cost
        base_idle = alpha * 0.5 + beta * 0.5 + gamma * 0.5
        for ii, L in enumerate(idle_logicals):
            row = len(logical_list) + ii
            if row >= N:
                break
            for P in range(N):
                centP = float(centrality.get(P, 0.0)) / cmax
                cost[row, P] = base_idle + 0.1 * centP  # mild penalty for taking central seats

        # any remaining padding rows (shouldn't exist since rows = N) -> uniform
        filled_rows = len(logical_list) + len(idle_logicals)
        for r in range(filled_rows, N):
            cost[r, :] = base_idle

        # ---- Hungarian assignment ----
        row_ind, col_ind = linear_sum_assignment(cost)

        new_map = [0] * N
        new_rev = [0] * N
        seen_phys = set()
        # rows: logical_list then idle_logicals then padding
        ordered_logicals = list(logical_list) + list(idle_logicals)
        # pad to length N with remaining ids
        used_log = set(ordered_logicals)
        for q in range(N):
            if q not in used_log:
                ordered_logicals.append(q)

        for r, p in zip(row_ind, col_ind):
            L = ordered_logicals[r]
            new_map[L] = int(p)
            new_rev[int(p)] = L
            seen_phys.add(int(p))

        # safety: if any duplicate (shouldn't be, Hungarian is a permutation), fall back
        if len(set(new_map)) != N:
            raise RuntimeError("non-injective hungarian result")

        self.mapping_dict = new_map
        self.reverse_mapping_dict = new_rev

    except Exception:
        try:
            from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
            md, rmd = generate_structure_aware_initial_mapping(
                self.access, self.backend, self.distance_matrix, self.num_qubits
            )
            self.mapping_dict = list(md)
            self.reverse_mapping_dict = list(rmd)
            if len(set(self.mapping_dict)) != N:
                self.mapping_dict = list(range(N))
                self.reverse_mapping_dict = list(range(N))
        except Exception:
            self.mapping_dict = list(range(N))
            self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)