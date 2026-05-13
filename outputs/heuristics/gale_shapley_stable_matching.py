def init_mapping(self):
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # 1. Collect logical qubits and their interaction load.
    logical_qubits = set()
    for _, qubits in self.access.items():
        for q in qubits:
            if 0 <= q < N:
                logical_qubits.add(q)

    load = defaultdict(int)
    activity = getattr(self, "logical_activity", None)
    if activity:
        for q in logical_qubits:
            load[q] = activity.get(q, 0) if isinstance(activity, dict) else activity[q]
    else:
        for _, qubits in self.access.items():
            if len(qubits) == 2:
                q1, q2 = qubits[0], qubits[1]
                load[q1] += 1
                load[q2] += 1

    # 2. Proposer preference: physical qubits ranked by centrality descending.
    centrality = getattr(self, "physical_centrality", {}) or {}
    phys_by_centrality = sorted(
        range(N), key=lambda p: (-float(centrality.get(p, 0.0)), p)
    )

    # 3. Receiver preference: logical qubits ranked by load descending.
    logicals_sorted = sorted(
        logical_qubits, key=lambda l: (-load.get(l, 0), l)
    )
    log_rank = {l: i for i, l in enumerate(logicals_sorted)}

    # 4. Gale-Shapley with logicals as proposers.
    next_idx = {l: 0 for l in logicals_sorted}
    engaged = {}  # physical -> logical
    free = list(logicals_sorted)

    while free:
        l = free.pop()
        while next_idx[l] < N:
            p = phys_by_centrality[next_idx[l]]
            next_idx[l] += 1
            current = engaged.get(p)
            if current is None:
                engaged[p] = l
                break
            if log_rank[l] < log_rank[current]:
                engaged[p] = l
                free.append(current)
                break
            # else physical rejects l; l proposes again next loop iter
        # if next_idx[l] == N, l stays unmatched (handled in back-fill)

    # 5. Apply GS matches.
    used_phys = set()
    for p, l in engaged.items():
        self.mapping_dict[l] = p
        self.reverse_mapping_dict[p] = l
        used_phys.add(p)

    # 6. Back-fill remaining logicals onto most-central free physicals.
    free_phys = [p for p in phys_by_centrality if p not in used_phys]
    fp_iter = iter(free_phys)
    for l in range(N):
        if self.mapping_dict[l] == -1:
            try:
                p = next(fp_iter)
            except StopIteration:
                break
            self.mapping_dict[l] = p
            self.reverse_mapping_dict[p] = l
            used_phys.add(p)

    # Hard identity fallback for any still-missing slot.
    if any(v == -1 for v in self.mapping_dict):
        leftover = [p for p in range(N) if p not in used_phys]
        lo_iter = iter(leftover)
        for l in range(N):
            if self.mapping_dict[l] == -1:
                p = next(lo_iter)
                self.mapping_dict[l] = p
                self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)