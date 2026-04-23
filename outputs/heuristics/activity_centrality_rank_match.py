def init_mapping(self):
    N = self.num_qubits

    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    if self.access2q is not None:
        active_logicals = set()
        for g, qs in self.access2q.items():
            if len(qs) == 2:
                active_logicals.add(qs[0])
                active_logicals.add(qs[1])
    else:
        active_logicals = set()
        for g, qs in self.access.items():
            if len(qs) == 2:
                active_logicals.add(qs[0])
                active_logicals.add(qs[1])

    activity = {}
    for q in active_logicals:
        if q in self.logical_activity:
            activity[q] = self.logical_activity[q]
        else:
            neigh = self.qubit_interaction_graph.get(q, {})
            activity[q] = sum(neigh.values())

    ranked_logicals = sorted(
        active_logicals,
        key=lambda q: (-activity.get(q, 0), q),
    )

    physical_ids = [p for p in range(N) if p in self.physical_centrality]
    if not physical_ids:
        physical_ids = list(range(N))

    ranked_physicals = sorted(
        physical_ids,
        key=lambda p: (-self.physical_centrality.get(p, 0.0), p),
    )

    used_physical = set()
    placed_logical = set()

    pair_count = min(len(ranked_logicals), len(ranked_physicals))
    for i in range(pair_count):
        lq = ranked_logicals[i]
        pq = ranked_physicals[i]
        if lq >= N or pq >= N:
            continue
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)
        placed_logical.add(lq)

    free_physicals = [p for p in range(N) if p not in used_physical]
    free_idx = 0
    for lq in range(N):
        if lq in placed_logical:
            continue
        if self.mapping_dict[lq] != -1:
            continue
        if lq not in used_physical and lq < N:
            chosen = lq if (lq not in used_physical) else None
        else:
            chosen = None
        if chosen is None or chosen in used_physical:
            while free_idx < len(free_physicals) and free_physicals[free_idx] in used_physical:
                free_idx += 1
            if free_idx >= len(free_physicals):
                break
            chosen = free_physicals[free_idx]
            free_idx += 1
        self.mapping_dict[lq] = chosen
        self.reverse_mapping_dict[chosen] = lq
        used_physical.add(chosen)

    if -1 in self.mapping_dict:
        remaining_phys = [p for p in range(N) if p not in used_physical]
        rp_iter = iter(remaining_phys)
        for lq in range(N):
            if self.mapping_dict[lq] == -1:
                try:
                    pq = next(rp_iter)
                except StopIteration:
                    break
                while pq in used_physical:
                    try:
                        pq = next(rp_iter)
                    except StopIteration:
                        pq = None
                        break
                if pq is None:
                    break
                self.mapping_dict[lq] = pq
                self.reverse_mapping_dict[pq] = lq
                used_physical.add(pq)

    if -1 in self.mapping_dict:
        used = set(p for p in self.mapping_dict if p != -1)
        spare = [p for p in range(N) if p not in used]
        si = 0
        for lq in range(N):
            if self.mapping_dict[lq] == -1:
                if si < len(spare):
                    self.mapping_dict[lq] = spare[si]
                    self.reverse_mapping_dict[spare[si]] = lq
                    si += 1

    if self.use_isl:
        try:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        except Exception:
            pass

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)