def init_mapping(self):
    import numpy as np
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [0] * N
    self.reverse_mapping_dict = [0] * N

    def fiedler_coords(adj_weights, nodes):
        # adj_weights: dict[node] -> dict[neighbor] -> weight; nodes: list of node ids
        n = len(nodes)
        coords = {v: 0.0 for v in nodes}
        if n <= 1:
            return coords
        idx = {v: i for i, v in enumerate(nodes)}
        # build connected components (over the subgraph induced by `nodes`)
        node_set = set(nodes)
        visited = set()
        components = []
        for v in nodes:
            if v in visited:
                continue
            stack = [v]
            comp = []
            visited.add(v)
            while stack:
                u = stack.pop()
                comp.append(u)
                for w in adj_weights.get(u, {}):
                    if w in node_set and w not in visited and adj_weights[u].get(w, 0) > 0:
                        visited.add(w)
                        stack.append(w)
            components.append(comp)

        for comp in components:
            cn = len(comp)
            if cn == 1:
                coords[comp[0]] = 0.0
                continue
            ci = {v: i for i, v in enumerate(comp)}
            L = np.zeros((cn, cn), dtype=float)
            for u in comp:
                for w, wt in adj_weights.get(u, {}).items():
                    if w in ci and wt > 0:
                        i_, j_ = ci[u], ci[w]
                        if i_ != j_:
                            L[i_, j_] -= float(wt)
                            L[i_, i_] += float(wt)
            try:
                L = (L + L.T) / 2.0
                eigvals, eigvecs = np.linalg.eigh(L)
                # second smallest (Fiedler) — eigvals are ascending
                fv = eigvecs[:, 1] if cn >= 2 else eigvecs[:, 0]
            except Exception:
                fv = np.array([adj_weights.get(v, {}) and sum(adj_weights[v].values()) or 0.0
                               for v in comp], dtype=float)
            for v in comp:
                coords[v] = float(fv[ci[v]])

        return coords

    try:
        # Logical interactions: prefer qubit_interaction_graph; fall back to self.access scan
        logical_adj = defaultdict(lambda: defaultdict(float))
        active_logicals = set()
        qig = getattr(self, "qubit_interaction_graph", None)
        access = getattr(self, "access", {}) or {}

        for gate_id, qubits in access.items():
            if len(qubits) == 2:
                a, b = qubits[0], qubits[1]
                if a == b:
                    continue
                active_logicals.add(a)
                active_logicals.add(b)
                w = 1.0
                if qig is not None:
                    try:
                        w = float(qig[a][b]) if qig[a][b] else 1.0
                    except Exception:
                        w = 1.0
                logical_adj[a][b] = max(logical_adj[a][b], w)
                logical_adj[b][a] = max(logical_adj[b][a], w)

        logical_nodes = sorted(active_logicals)

        # Hardware (physical) graph over all N qubits
        physical_adj = defaultdict(lambda: defaultdict(float))
        backend = getattr(self, "backend", {}) or {}
        for u in range(N):
            for v in backend.get(u, []):
                if 0 <= v < N and u != v:
                    physical_adj[u][v] = 1.0
                    physical_adj[v][u] = 1.0
        physical_nodes = list(range(N))

        log_coords = fiedler_coords(logical_adj, logical_nodes)
        phys_coords = fiedler_coords(physical_adj, physical_nodes)

        # Tie-break logicals by activity (more active first), physicals by centrality
        log_activity = getattr(self, "logical_activity", {}) or {}
        phys_central = getattr(self, "physical_centrality", {}) or {}

        sorted_logicals = sorted(
            logical_nodes,
            key=lambda q: (log_coords.get(q, 0.0), -float(log_activity.get(q, 0)), q),
        )
        sorted_physicals = sorted(
            physical_nodes,
            key=lambda p: (phys_coords.get(p, 0.0), -float(phys_central.get(p, 0.0)), p),
        )

        used_phys = set()
        assigned_log = set()

        # Pair by sorted Fiedler position
        for i, lq in enumerate(sorted_logicals):
            if i >= len(sorted_physicals):
                break
            pq = sorted_physicals[i]
            if 0 <= lq < N and pq not in used_phys:
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_phys.add(pq)
                assigned_log.add(lq)

        # Back-fill remaining logicals onto most-central unused physicals
        remaining_phys = [p for p in sorted(
            physical_nodes,
            key=lambda p: (-float(phys_central.get(p, 0.0)), p),
        ) if p not in used_phys]
        remaining_log = [l for l in range(N) if l not in assigned_log]

        rp_iter = iter(remaining_phys)
        for lq in remaining_log:
            placed = False
            for pq in rp_iter:
                if pq not in used_phys:
                    self.mapping_dict[lq] = pq
                    self.reverse_mapping_dict[pq] = lq
                    used_phys.add(pq)
                    assigned_log.add(lq)
                    placed = True
                    break
            if not placed:
                # Identity fallback for this logical
                if lq < N and lq not in used_phys:
                    self.mapping_dict[lq] = lq
                    self.reverse_mapping_dict[lq] = lq
                    used_phys.add(lq)
                    assigned_log.add(lq)

        # Final safety: if anything is still inconsistent, repair via identity over free slots
        if len(set(self.mapping_dict)) != N:
            used_phys = set()
            for lq in range(N):
                pq = self.mapping_dict[lq]
                if pq in used_phys or not (0 <= pq < N):
                    self.mapping_dict[lq] = -1
                else:
                    used_phys.add(pq)
            free = [p for p in range(N) if p not in used_phys]
            fi = 0
            for lq in range(N):
                if self.mapping_dict[lq] == -1:
                    self.mapping_dict[lq] = free[fi]
                    fi += 1
            for lq in range(N):
                self.reverse_mapping_dict[self.mapping_dict[lq]] = lq

    except Exception:
        # Hard fallback: identity mapping
        for q in range(N):
            self.mapping_dict[q] = q
            self.reverse_mapping_dict[q] = q

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)