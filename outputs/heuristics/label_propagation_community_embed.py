def init_mapping(self):
    from collections import defaultdict, deque
    import random as _random

    N = self.num_qubits
    self.mapping_dict = [None] * N
    self.reverse_mapping_dict = [None] * N

    qig = self.qubit_interaction_graph
    active_logicals = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            active_logicals.add(qubits[0])
            active_logicals.add(qubits[1])
    active_logicals = {q for q in active_logicals if 0 <= q < N}

    def label_propagation(nodes, graph, max_iter=20, seed=0):
        rng = _random.Random(seed)
        labels = {n: n for n in nodes}
        node_list = list(nodes)
        for _ in range(max_iter):
            rng.shuffle(node_list)
            changed = False
            for n in node_list:
                neigh = graph.get(n, {})
                if not neigh:
                    continue
                score = defaultdict(float)
                for m, w in neigh.items():
                    if m in labels:
                        score[labels[m]] += float(w)
                if not score:
                    continue
                best = max(score.values())
                top = [lab for lab, s in score.items() if s == best]
                new_label = top[0] if len(top) == 1 else rng.choice(top)
                if new_label != labels[n]:
                    labels[n] = new_label
                    changed = True
            if not changed:
                break
        return labels

    labels = label_propagation(active_logicals, qig)

    communities = defaultdict(list)
    for node, lab in labels.items():
        communities[lab].append(node)

    def intra_weight(members):
        s = 0.0
        mset = set(members)
        for u in members:
            for v, w in qig.get(u, {}).items():
                if v in mset:
                    s += float(w)
        return s * 0.5

    community_list = list(communities.values())
    community_list.sort(key=lambda c: (-intra_weight(c), -len(c)))

    for i in range(len(community_list)):
        community_list[i] = sorted(
            community_list[i],
            key=lambda q: -self.logical_activity.get(q, 0)
        )

    centrality = self.physical_centrality
    phys_sorted = sorted(range(N), key=lambda p: -centrality.get(p, 0.0))

    backend = self.backend

    def bfs_region(seed, size, used):
        region = []
        visited = {seed}
        dq = deque([seed])
        while dq and len(region) < size:
            cur = dq.popleft()
            if cur in used:
                continue
            region.append(cur)
            neighbors = backend.get(cur, []) if isinstance(backend, dict) else []
            if not isinstance(backend, dict):
                try:
                    neighbors = list(backend[cur])
                except Exception:
                    neighbors = []
            pairs = []
            for nb in neighbors:
                if nb not in visited and nb not in used:
                    pairs.append((centrality.get(nb, 0.0), nb))
            pairs.sort(key=lambda x: -x[0])
            for _, nb in pairs:
                visited.add(nb)
                dq.append(nb)
        return region

    used_phys = set()
    assigned_log = set()

    for community in community_list:
        if not community:
            continue
        seed = None
        for p in phys_sorted:
            if p not in used_phys:
                seed = p
                break
        if seed is None:
            break
        region = bfs_region(seed, len(community), used_phys)
        if len(region) < len(community):
            extra_needed = len(community) - len(region)
            for p in phys_sorted:
                if p not in used_phys and p not in region:
                    region.append(p)
                    extra_needed -= 1
                    if extra_needed == 0:
                        break
        for log_q, phys_q in zip(community, region):
            if log_q in assigned_log or phys_q in used_phys:
                continue
            if 0 <= log_q < N and 0 <= phys_q < N:
                self.mapping_dict[log_q] = phys_q
                self.reverse_mapping_dict[phys_q] = log_q
                assigned_log.add(log_q)
                used_phys.add(phys_q)

    remaining_log = [q for q in range(N) if q not in assigned_log]
    remaining_phys = [p for p in range(N) if p not in used_phys]
    for log_q, phys_q in zip(remaining_log, remaining_phys):
        self.mapping_dict[log_q] = phys_q
        self.reverse_mapping_dict[phys_q] = log_q
        assigned_log.add(log_q)
        used_phys.add(phys_q)

    for q in range(N):
        if self.mapping_dict[q] is None:
            for p in range(N):
                if p not in used_phys:
                    self.mapping_dict[q] = p
                    self.reverse_mapping_dict[p] = q
                    used_phys.add(p)
                    break

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)