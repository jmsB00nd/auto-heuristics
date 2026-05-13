def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    adj = defaultdict(set)
    weight = defaultdict(lambda: defaultdict(float))

    qig = getattr(self, "qubit_interaction_graph", None)
    if qig is not None:
        for u, nbrs in qig.items():
            if u < 0 or u >= N:
                continue
            for v, w in nbrs.items():
                if v < 0 or v >= N or v == u:
                    continue
                if w <= 0:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                weight[u][v] = float(w)
                weight[v][u] = float(w)

    access = getattr(self, "access", {}) or {}
    for _, qubits in access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            if 0 <= a < N and 0 <= b < N:
                if b not in adj[a]:
                    adj[a].add(b)
                    adj[b].add(a)
                    weight[a][b] = max(weight[a][b], 1.0)
                    weight[b][a] = max(weight[b][a], 1.0)

    active = set(range(N))
    work_adj = {v: set(adj[v]) for v in active}
    work_w = {v: dict(weight[v]) for v in active}

    order = []
    remaining = set(active)

    while remaining:
        best_v = None
        best_fill = None
        best_wdeg = None
        for v in remaining:
            nbrs = work_adj[v]
            if not nbrs:
                fill = 0
                wdeg = 0.0
            else:
                fill = 0
                nbr_list = list(nbrs)
                for i in range(len(nbr_list)):
                    a = nbr_list[i]
                    a_adj = work_adj[a]
                    for j in range(i + 1, len(nbr_list)):
                        b = nbr_list[j]
                        if b not in a_adj:
                            fill += 1
                wdeg = sum(work_w[v].get(u, 0.0) for u in nbrs)
            if (best_v is None
                or fill < best_fill
                or (fill == best_fill and wdeg < best_wdeg)
                or (fill == best_fill and wdeg == best_wdeg and v < best_v)):
                best_v = v
                best_fill = fill
                best_wdeg = wdeg

        v = best_v
        nbrs = list(work_adj[v])
        for i in range(len(nbrs)):
            a = nbrs[i]
            for j in range(i + 1, len(nbrs)):
                b = nbrs[j]
                if b not in work_adj[a]:
                    work_adj[a].add(b)
                    work_adj[b].add(a)
                    w_new = max(work_w[a].get(v, 0.0), work_w[b].get(v, 0.0))
                    work_w[a][b] = w_new
                    work_w[b][a] = w_new
        for u in nbrs:
            work_adj[u].discard(v)
            if v in work_w[u]:
                del work_w[u][v]
        del work_adj[v]
        del work_w[v]
        order.append(v)
        remaining.discard(v)

    priority = list(reversed(order))
    seen = set()
    dedup_priority = []
    for q in priority:
        if q not in seen and 0 <= q < N:
            dedup_priority.append(q)
            seen.add(q)
    for q in range(N):
        if q not in seen:
            dedup_priority.append(q)
            seen.add(q)

    centrality = getattr(self, "physical_centrality", None) or {}
    centrals = sorted(range(N), key=lambda p: (-float(centrality.get(p, 0.0)), p))

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N
    used_phys = set()
    assigned_log = set()

    for log_q, phys_q in zip(dedup_priority, centrals):
        if log_q in assigned_log or phys_q in used_phys:
            continue
        self.mapping_dict[log_q] = phys_q
        self.reverse_mapping_dict[phys_q] = log_q
        used_phys.add(phys_q)
        assigned_log.add(log_q)

    free_phys = [p for p in range(N) if p not in used_phys]
    fp_idx = 0
    for log_q in range(N):
        if self.mapping_dict[log_q] == -1:
            while fp_idx < len(free_phys) and free_phys[fp_idx] in used_phys:
                fp_idx += 1
            if fp_idx >= len(free_phys):
                break
            p = free_phys[fp_idx]
            fp_idx += 1
            self.mapping_dict[log_q] = p
            self.reverse_mapping_dict[p] = log_q
            used_phys.add(p)

    for log_q in range(N):
        if self.mapping_dict[log_q] == -1:
            for p in range(N):
                if p not in used_phys:
                    self.mapping_dict[log_q] = p
                    self.reverse_mapping_dict[p] = log_q
                    used_phys.add(p)
                    break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)