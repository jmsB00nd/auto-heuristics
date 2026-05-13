def init_mapping(self):
    from collections import defaultdict, deque

    N = self.num_qubits
    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    try:
        access = self.access or {}
        gate_ids = sorted(access.keys())

        last_writer = {}
        predecessors = defaultdict(list)
        successors = defaultdict(list)
        for gid in gate_ids:
            qubits = access[gid]
            preds = set()
            for q in qubits:
                if q in last_writer:
                    preds.add(last_writer[q])
            for p in preds:
                predecessors[gid].append(p)
                successors[p].append(gid)
            for q in qubits:
                last_writer[q] = gid

        cl = {}
        try:
            from src.graph.graph import compute_longest_chain_lengths
            raw = compute_longest_chain_lengths(successors, predecessors)
            if isinstance(raw, list):
                for gid in gate_ids:
                    if gid < len(raw):
                        cl[gid] = raw[gid]
            elif isinstance(raw, dict):
                cl = dict(raw)
        except Exception:
            cl = {}

        if not cl:
            in_deg = {g: len(predecessors[g]) for g in gate_ids}
            q = deque([g for g in gate_ids if in_deg[g] == 0])
            topo = []
            while q:
                g = q.popleft()
                topo.append(g)
                for s in successors[g]:
                    in_deg[s] -= 1
                    if in_deg[s] == 0:
                        q.append(s)
            for g in topo:
                cl[g] = 1 + max((cl[p] for p in predecessors[g]), default=0)

        chain_gates = []
        if cl:
            cur = max(cl, key=lambda g: cl[g])
            while cur is not None:
                chain_gates.append(cur)
                target = cl[cur] - 1
                nxt = None
                for p in predecessors[cur]:
                    if cl.get(p, -1) == target:
                        nxt = p
                        break
                cur = nxt
            chain_gates.reverse()

        critical = []
        seen = set()
        for gid in chain_gates:
            for lq in access.get(gid, []):
                if lq not in seen and 0 <= lq < N:
                    seen.add(lq)
                    critical.append(lq)

        K = len(critical)

        def find_path(start, target_len):
            best = [start]
            stack = [(start, [start], {start})]
            iters = 0
            cap = 20000
            while stack and iters < cap:
                iters += 1
                node, path, visited = stack.pop()
                if len(path) > len(best):
                    best = list(path)
                    if len(best) >= target_len:
                        return best
                nbrs = list(self.backend.get(node, []))
                nbrs.sort(key=lambda x: self.physical_centrality.get(x, 0.0))
                for nb in nbrs:
                    if nb not in visited:
                        stack.append((nb, path + [nb], visited | {nb}))
            return best

        hw_path = []
        if K > 0:
            seeds = sorted(range(N), key=lambda p: -self.physical_centrality.get(p, 0.0))
            for s in seeds[:min(N, 8)]:
                cand = find_path(s, K)
                if len(cand) > len(hw_path):
                    hw_path = cand
                if len(hw_path) >= K:
                    break

        mapping = [None] * N
        reverse = [None] * N
        used_phys = set()

        for i, lq in enumerate(critical):
            if i < len(hw_path):
                pq = hw_path[i]
                if mapping[lq] is None and pq not in used_phys:
                    mapping[lq] = pq
                    reverse[pq] = lq
                    used_phys.add(pq)

        all_logicals = set()
        for qs in access.values():
            for q in qs:
                if 0 <= q < N:
                    all_logicals.add(q)
        assigned = {lq for lq in range(N) if mapping[lq] is not None}
        remaining_logicals = sorted(
            (set(range(N)) - assigned),
            key=lambda q: (-(1 if q in all_logicals else 0),
                           -self.logical_activity.get(q, 0)),
        )
        unused_phys = sorted(
            set(range(N)) - used_phys,
            key=lambda p: -self.physical_centrality.get(p, 0.0),
        )
        for lq, pq in zip(remaining_logicals, unused_phys):
            mapping[lq] = pq
            reverse[pq] = lq
            used_phys.add(pq)

        leftover_phys = [p for p in range(N) if p not in used_phys]
        for lq in range(N):
            if mapping[lq] is None and leftover_phys:
                pq = leftover_phys.pop(0)
                mapping[lq] = pq
                reverse[pq] = lq

        if all(x is not None for x in mapping) and len(set(mapping)) == N:
            self.mapping_dict = mapping
            self.reverse_mapping_dict = reverse
        else:
            self.mapping_dict = list(range(N))
            self.reverse_mapping_dict = list(range(N))
    except Exception:
        self.mapping_dict = list(range(N))
        self.reverse_mapping_dict = list(range(N))

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)