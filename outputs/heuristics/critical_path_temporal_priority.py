def init_mapping(self):
    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    earliest_depth = {}
    chain = {}
    finish_depth = {}
    logical_qubits = set()

    gate_ids = sorted(self.access.keys())
    for idx, gid in enumerate(gate_ids):
        qubits = self.access[gid]
        for q in qubits:
            logical_qubits.add(q)
            if q not in earliest_depth:
                earliest_depth[q] = idx
                chain[q] = 1
                finish_depth[q] = idx
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            new_chain = max(chain[a], chain[b]) + 1
            chain[a] = new_chain
            chain[b] = new_chain
            finish_depth[a] = idx
            finish_depth[b] = idx

    activity = getattr(self, "logical_activity", {})

    def priority_key(q):
        return (earliest_depth.get(q, 10**9),
                -chain.get(q, 0),
                -activity.get(q, 0),
                q)

    priority_logicals = sorted(logical_qubits, key=priority_key)

    centrality = getattr(self, "physical_centrality", {})
    physical_order = sorted(range(N),
                            key=lambda p: (-centrality.get(p, 0.0), p))

    used_phys = set()
    phys_iter_idx = 0

    for lq in priority_logicals:
        if lq < 0 or lq >= N:
            continue
        while phys_iter_idx < len(physical_order) and physical_order[phys_iter_idx] in used_phys:
            phys_iter_idx += 1
        if phys_iter_idx >= len(physical_order):
            break
        p = physical_order[phys_iter_idx]
        self.mapping_dict[lq] = p
        self.reverse_mapping_dict[p] = lq
        used_phys.add(p)
        phys_iter_idx += 1

    unmapped_logicals = [q for q in range(N) if self.mapping_dict[q] == -1]
    remaining_phys = [p for p in physical_order if p not in used_phys]

    for lq in unmapped_logicals:
        if not remaining_phys:
            break
        p = remaining_phys.pop(0)
        self.mapping_dict[lq] = p
        self.reverse_mapping_dict[p] = lq
        used_phys.add(p)

    if any(m == -1 for m in self.mapping_dict):
        free_phys = [p for p in range(N) if p not in used_phys]
        for lq in range(N):
            if self.mapping_dict[lq] == -1:
                if not free_phys:
                    break
                p = free_phys.pop(0)
                self.mapping_dict[lq] = p
                self.reverse_mapping_dict[p] = lq
                used_phys.add(p)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)