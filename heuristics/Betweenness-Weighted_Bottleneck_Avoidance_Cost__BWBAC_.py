def qlosure_poly_heuristic(self, swap_gate):
    from collections import deque

    # ── Lazily compute betweenness centrality (Brandes, unweighted) and
    #    a path-bottleneck matrix, both cached on the instance. ──────────────
    if not hasattr(self, '_bwbac_path_matrix'):
        nodes = list(self.backend.keys())
        n     = len(nodes)
        bc    = {v: 0.0 for v in nodes}

        # Brandes algorithm – O(V·E) for unweighted undirected graphs
        for s in nodes:
            stack   = []
            pred    = {v: [] for v in nodes}
            sigma   = {v: 0.0 for v in nodes};  sigma[s] = 1.0
            d       = {v: -1  for v in nodes};  d[s]     = 0
            q       = deque([s])
            while q:
                v = q.popleft()
                stack.append(v)
                for w in self.backend[v]:
                    if d[w] < 0:
                        q.append(w); d[w] = d[v] + 1
                    if d[w] == d[v] + 1:
                        sigma[w] += sigma[v];  pred[w].append(v)
            delta = {v: 0.0 for v in nodes}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    bc[w] += delta[w]

        # Normalise to [0, 1]
        if n > 2:
            scale = 1.0 / ((n - 1) * (n - 2))
            for v in bc:
                bc[v] *= scale
        self._bwbac_bc = bc

        # Precompute path-bottleneck matrix:
        #   bc_path[s][t] = mean BC of strictly intermediate nodes on the
        #                   shortest path s → t  (0 when d(s,t) ≤ 1)
        n_max   = max(nodes) + 1
        bc_path = [[0.0] * n_max for _ in range(n_max)]

        for s in nodes:
            pred_map = {v: -1 for v in nodes}
            d        = {v: -1 for v in nodes};  d[s] = 0
            q        = deque([s])
            while q:
                v = q.popleft()
                for w in self.backend[v]:
                    if d[w] < 0:
                        d[w] = d[v] + 1;  pred_map[w] = v;  q.append(w)

            for t in nodes:
                if t == s or d[t] <= 1:          # distance 0 or 1 → no intermediates
                    continue
                total  = 0.0
                cursor = pred_map[t]
                while cursor != s:               # walk path backwards, skip endpoints
                    total += bc[cursor]
                    cursor = pred_map[cursor]
                bc_path[s][t] = total / (d[t] - 1)   # mean intermediate BC

        self._bwbac_path_matrix = bc_path

    bc_path = self._bwbac_path_matrix

    # ── BWBAC cost ────────────────────────────────────────────────────────────
    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # Front layer: distance × (1 + mean-BC bottleneck), weighted by gate deps
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        deps       = self.dag_dependencies_count[g]
        dist       = self.distance_matrix[Q1][Q2]
        bottleneck = 1.0 + bc_path[Q1][Q2]          # ≥ 1; high-BC paths amplified
        f_cost    += (deps + 1) * dist * bottleneck

    # Extended layer: same, decayed by look-ahead depth
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        deps         = self.dag_dependencies_count[g]
        dist         = self.distance_matrix[Q1][Q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        bottleneck   = 1.0 + bc_path[Q1][Q2]
        e_cost      += (deps + 1) * dist * bottleneck / layer_factor

    W = 1.0
    H = max_decay * (
        f_cost / front_layer_size
        + W * (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )
    return H