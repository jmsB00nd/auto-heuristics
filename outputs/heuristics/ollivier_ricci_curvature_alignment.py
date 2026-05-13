def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        linear_sum_assignment = None

    N = self.num_qubits

    # ---- collect logical 2q interactions ----
    logical_adj = defaultdict(set)
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            logical_qubits.add(a); logical_qubits.add(b)
            if a != b:
                logical_adj[a].add(b)
                logical_adj[b].add(a)

    # ---- BFS distance on logical graph ----
    def bfs_dist(src, adj, nodes):
        dist = {src: 0}
        dq = deque([src])
        while dq:
            u = dq.popleft()
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    dq.append(v)
        for n in nodes:
            if n not in dist:
                dist[n] = 10**6
        return dist

    # ---- W1 via Hungarian on small bipartite cost ----
    def emd(p_keys, p_w, q_keys, q_w, dmat_lookup):
        # balance masses by replication-free LP via Hungarian on a square padded matrix
        # Build matrix of pairwise distances
        n_p, n_q = len(p_keys), len(q_keys)
        if n_p == 0 or n_q == 0:
            return 0.0
        # Use Sinkhorn-free approximation: since masses are uniform 1/deg, W1 reduces to
        # min over assignments of average distance when |p|==|q|; otherwise pad with zeros.
        size = max(n_p, n_q)
        cost = np.zeros((size, size), dtype=float)
        for i in range(size):
            for j in range(size):
                if i < n_p and j < n_q:
                    cost[i, j] = dmat_lookup(p_keys[i], q_keys[j])
                else:
                    cost[i, j] = 0.0
        if linear_sum_assignment is not None:
            try:
                ri, ci = linear_sum_assignment(cost)
                total = float(cost[ri, ci].sum())
                return total / float(size)
            except Exception:
                pass
        # greedy fallback
        used_c = set()
        total = 0.0
        for i in range(size):
            best = None; bv = float('inf')
            for j in range(size):
                if j in used_c: continue
                if cost[i, j] < bv:
                    bv = cost[i, j]; best = j
            used_c.add(best); total += bv
        return total / float(size)

    # ---- curvature on logical graph ----
    log_nodes = sorted(logical_qubits)
    log_dist_cache = {}
    def log_dist(u, v):
        if u == v: return 0
        if u not in log_dist_cache:
            log_dist_cache[u] = bfs_dist(u, logical_adj, log_nodes)
        return log_dist_cache[u].get(v, 10**6)

    log_curv_node = {n: 0.0 for n in log_nodes}
    log_edges_seen = defaultdict(list)
    seen_edges = set()
    for u in log_nodes:
        for v in logical_adj[u]:
            if (v, u) in seen_edges or (u, v) in seen_edges:
                continue
            seen_edges.add((u, v))
            nu = list(logical_adj[u]); nv = list(logical_adj[v])
            if not nu or not nv:
                kappa = 0.0
            else:
                wu = [1.0 / len(nu)] * len(nu)
                wv = [1.0 / len(nv)] * len(nv)
                w1 = emd(nu, wu, nv, wv, log_dist)
                d = log_dist(u, v)
                if d <= 0: d = 1
                kappa = 1.0 - w1 / d
            log_edges_seen[u].append(kappa)
            log_edges_seen[v].append(kappa)
    log_mean_curv = {}
    for n in log_nodes:
        if log_edges_seen[n]:
            log_mean_curv[n] = float(np.mean(log_edges_seen[n]))
        else:
            log_mean_curv[n] = 0.0

    # ---- curvature on hardware graph ----
    phys_nodes = list(range(N))
    def phys_dist(u, v):
        if u == v: return 0
        d = self.distance_matrix[u][v]
        if d <= 0:
            return 10**6
        return d

    phys_edges_seen = defaultdict(list)
    seen_edges_h = set()
    for u in phys_nodes:
        for v in self.backend.get(u, ()):
            if (v, u) in seen_edges_h or (u, v) in seen_edges_h:
                continue
            seen_edges_h.add((u, v))
            nu = list(self.backend.get(u, ()))
            nv = list(self.backend.get(v, ()))
            if not nu or not nv:
                kappa = 0.0
            else:
                wu = [1.0 / len(nu)] * len(nu)
                wv = [1.0 / len(nv)] * len(nv)
                w1 = emd(nu, wu, nv, wv, phys_dist)
                d = phys_dist(u, v)
                if d <= 0: d = 1
                kappa = 1.0 - w1 / d
            phys_edges_seen[u].append(kappa)
            phys_edges_seen[v].append(kappa)
    phys_mean_curv = {}
    for n in phys_nodes:
        if phys_edges_seen[n]:
            phys_mean_curv[n] = float(np.mean(phys_edges_seen[n]))
        else:
            phys_mean_curv[n] = 0.0

    # ---- Hungarian matching on curvature alignment ----
    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    if log_nodes:
        # Order physical candidates by centrality (more central first)
        try:
            phys_order = sorted(phys_nodes,
                                key=lambda p: -float(self.physical_centrality.get(p, 0.0)))
        except Exception:
            phys_order = list(phys_nodes)
        L = len(log_nodes)
        P = len(phys_order)
        size = max(L, P)
        cost = np.full((size, size), 1e6, dtype=float)
        for i, lq in enumerate(log_nodes):
            for j, pq in enumerate(phys_order):
                cost[i, j] = abs(log_mean_curv[lq] - phys_mean_curv[pq])
        # pad rows/cols with a constant so Hungarian still works
        for i in range(L, size):
            for j in range(size):
                cost[i, j] = 0.0
        for j in range(P, size):
            for i in range(size):
                cost[i, j] = 0.0

        assigned_log = {}
        used_phys = set()
        if linear_sum_assignment is not None:
            try:
                ri, ci = linear_sum_assignment(cost)
                for i, j in zip(ri, ci):
                    if i < L and j < P:
                        lq = log_nodes[i]
                        pq = phys_order[j]
                        assigned_log[lq] = pq
                        used_phys.add(pq)
            except Exception:
                pass
        if not assigned_log:
            # greedy fallback by curvature proximity
            remaining = set(phys_order)
            for lq in sorted(log_nodes, key=lambda x: -log_mean_curv[x]):
                best = None; bv = float('inf')
                for pq in remaining:
                    v = abs(log_mean_curv[lq] - phys_mean_curv[pq])
                    if v < bv:
                        bv = v; best = pq
                if best is None:
                    break
                assigned_log[lq] = best
                remaining.discard(best)
                used_phys.add(best)

        # Write assignments
        for lq, pq in assigned_log.items():
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq

        # Back-fill remaining logicals (those not in log_nodes or unassigned) to unused physicals
        all_logical = set(range(N))
        placed_logicals = set(assigned_log.keys())
        # Logicals still needing a physical: any logical id whose current mapping collides with used_phys
        to_place = [l for l in range(N) if l not in placed_logicals]
        free_phys = [p for p in phys_order if p not in used_phys]
        # also include any physicals not in phys_order (shouldn't happen, but safe)
        for p in range(N):
            if p not in used_phys and p not in free_phys:
                free_phys.append(p)
        for lq, pq in zip(to_place, free_phys):
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq
            used_phys.add(pq)

        # Final repair: ensure injection by detecting duplicates and reassigning
        seen = {}
        dup_logicals = []
        for lq, pq in enumerate(self.mapping_dict):
            if pq in seen:
                dup_logicals.append(lq)
            else:
                seen[pq] = lq
        free_left = [p for p in range(N) if p not in seen]
        for lq, pq in zip(dup_logicals, free_left):
            self.mapping_dict[lq] = pq
            self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)