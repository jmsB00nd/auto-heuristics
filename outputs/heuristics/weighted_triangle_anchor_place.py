def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits

    backend_adj = defaultdict(set)
    for (a, b) in self.backend_connections:
        backend_adj[a].add(b)
        backend_adj[b].add(a)

    qig = self.qubit_interaction_graph

    logical_triangles = []
    for q1 in list(qig.keys()):
        neigh = sorted(n for n in qig[q1] if n > q1)
        for i, q2 in enumerate(neigh):
            adj_q2 = qig.get(q2, {})
            for q3 in neigh[i + 1:]:
                if q3 in adj_q2:
                    w = (qig[q1].get(q2, 0)
                         + adj_q2.get(q3, 0)
                         + qig[q1].get(q3, 0))
                    logical_triangles.append((w, (q1, q2, q3)))
    logical_triangles.sort(key=lambda x: -x[0])

    physical_triangles = []
    for p1 in sorted(backend_adj.keys()):
        neigh = sorted(n for n in backend_adj[p1] if n > p1)
        for i, p2 in enumerate(neigh):
            adj_p2 = backend_adj[p2]
            for p3 in neigh[i + 1:]:
                if p3 in adj_p2:
                    cent = (self.physical_centrality.get(p1, 0.0)
                            + self.physical_centrality.get(p2, 0.0)
                            + self.physical_centrality.get(p3, 0.0))
                    physical_triangles.append((cent, (p1, p2, p3)))
    physical_triangles.sort(key=lambda x: -x[0])

    mapped_logical = set()
    used_physical = set()
    assignments = {}

    pt_idx = 0
    for _, (lq1, lq2, lq3) in logical_triangles:
        if (lq1 in mapped_logical
                or lq2 in mapped_logical
                or lq3 in mapped_logical):
            continue
        if lq1 >= N or lq2 >= N or lq3 >= N:
            continue
        chosen = None
        while pt_idx < len(physical_triangles):
            _, trip = physical_triangles[pt_idx]
            if (trip[0] not in used_physical
                    and trip[1] not in used_physical
                    and trip[2] not in used_physical):
                chosen = trip
                break
            pt_idx += 1
        if chosen is None:
            break
        log_sorted = sorted([lq1, lq2, lq3],
                            key=lambda q: -self.logical_activity.get(q, 0))
        phys_sorted = sorted(list(chosen),
                             key=lambda p: -self.physical_centrality.get(p, 0.0))
        for lq, pq in zip(log_sorted, phys_sorted):
            assignments[lq] = pq
            mapped_logical.add(lq)
            used_physical.add(pq)
        pt_idx += 1

    all_logical = set()
    for qs in self.access.values():
        for q in qs:
            if 0 <= q < N:
                all_logical.add(q)

    remaining_logical = sorted(
        (q for q in all_logical if q not in mapped_logical),
        key=lambda q: -self.logical_activity.get(q, 0)
    )
    remaining_physical = sorted(
        (p for p in range(N) if p not in used_physical),
        key=lambda p: -self.physical_centrality.get(p, 0.0)
    )
    for lq, pq in zip(remaining_logical, remaining_physical):
        assignments[lq] = pq
        mapped_logical.add(lq)
        used_physical.add(pq)

    new_mapping = [-1] * N
    for lq, pq in assignments.items():
        if 0 <= lq < N and 0 <= pq < N and new_mapping[lq] == -1:
            new_mapping[lq] = pq

    seen_phys = set(p for p in new_mapping if p != -1)
    leftover = [p for p in range(N) if p not in seen_phys]
    li = 0
    for i in range(N):
        if new_mapping[i] == -1:
            new_mapping[i] = leftover[li]
            li += 1

    self.mapping_dict = new_mapping
    self.reverse_mapping_dict = [0] * N
    for lq in range(N):
        self.reverse_mapping_dict[self.mapping_dict[lq]] = lq

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)