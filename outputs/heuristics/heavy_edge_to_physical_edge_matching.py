def init_mapping(self):
    N = self.num_qubits
    mapping = [None] * N
    reverse = [None] * N
    used_phys = set()

    phys_adj = {i: set() for i in range(N)}
    for a, b in self.backend_connections:
        if a < N and b < N:
            phys_adj[a].add(b)
            phys_adj[b].add(a)

    seen_pairs = set()
    logical_edges = []
    for q1, nbrs in self.qubit_interaction_graph.items():
        if q1 >= N:
            continue
        for q2, w in nbrs.items():
            if q2 >= N or q1 == q2:
                continue
            pair = (q1, q2) if q1 < q2 else (q2, q1)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            logical_edges.append((w, pair[0], pair[1]))
    logical_edges.sort(key=lambda x: -x[0])

    phys_edges = [tuple(e) for e in self.backend_connections
                  if e[0] < N and e[1] < N]
    phys_edges.sort(key=lambda e: -(self.physical_centrality.get(e[0], 0.0)
                                    + self.physical_centrality.get(e[1], 0.0)))

    def do_place(lq, pq):
        mapping[lq] = pq
        reverse[pq] = lq
        used_phys.add(pq)

    for w, lq1, lq2 in logical_edges:
        m1, m2 = mapping[lq1], mapping[lq2]
        if m1 is not None and m2 is not None:
            continue
        if m1 is None and m2 is None:
            chosen = None
            for pa, pb in phys_edges:
                if pa not in used_phys and pb not in used_phys:
                    chosen = (pa, pb)
                    break
            if chosen is None:
                continue
            pa, pb = chosen
            ca = self.physical_centrality.get(pa, 0.0)
            cb = self.physical_centrality.get(pb, 0.0)
            la = self.logical_activity.get(lq1, 0)
            lb = self.logical_activity.get(lq2, 0)
            if (la >= lb) == (ca >= cb):
                do_place(lq1, pa); do_place(lq2, pb)
            else:
                do_place(lq1, pb); do_place(lq2, pa)
        else:
            if m1 is not None:
                anchor, other = m1, lq2
            else:
                anchor, other = m2, lq1
            best, best_score = None, -1.0
            for nb in phys_adj.get(anchor, ()):
                if nb in used_phys:
                    continue
                sc = self.physical_centrality.get(nb, 0.0)
                if sc > best_score:
                    best_score = sc
                    best = nb
            if best is not None:
                do_place(other, best)

    D = len(self.distance_matrix)
    remaining = [q for q in range(N)
                 if mapping[q] is None and q in self.qubit_interaction_graph]
    remaining.sort(key=lambda q: -self.logical_activity.get(q, 0))
    for lq in remaining:
        partners = [(mapping[nb], w)
                    for nb, w in self.qubit_interaction_graph.get(lq, {}).items()
                    if nb < N and mapping[nb] is not None]
        if not partners:
            continue
        best_p, best_cost = None, float('inf')
        for p in range(D):
            if p in used_phys:
                continue
            cost = 0.0
            for pp, w in partners:
                if pp < D:
                    cost += w * self.distance_matrix[p][pp]
                else:
                    cost += w * 1e9
            if cost < best_cost:
                best_cost = cost
                best_p = p
        if best_p is not None:
            do_place(lq, best_p)

    unused = [p for p in range(N) if p not in used_phys]
    unused_set = set(unused)
    for lq in range(N):
        if mapping[lq] is not None:
            continue
        if lq in unused_set:
            do_place(lq, lq)
            unused_set.discard(lq)
        else:
            for p in list(unused_set):
                do_place(lq, p)
                unused_set.discard(p)
                break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    if self.use_isl:
        try:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        except Exception:
            self.isl_mapping = None

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)