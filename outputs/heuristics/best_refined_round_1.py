def init_mapping(self):
    from collections import defaultdict, deque
    import random as _rnd

    N = self.num_qubits

    self.mapping_dict = list(range(N))
    self.reverse_mapping_dict = list(range(N))

    qig = defaultdict(lambda: defaultdict(float))
    logical_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            if 0 <= q1 < N and 0 <= q2 < N and q1 != q2:
                qig[q1][q2] += 1.0
                qig[q2][q1] += 1.0
                logical_nodes.add(q1)
                logical_nodes.add(q2)

    if not logical_nodes:
        assert len(set(self.mapping_dict)) == len(self.mapping_dict)
        return

    labels = {q: q for q in logical_nodes}
    rng = _rnd.Random(42)
    nodes = list(logical_nodes)
    for _it in range(20):
        rng.shuffle(nodes)
        changed = False
        for q in nodes:
            nbrs = qig.get(q, {})
            if not nbrs:
                continue
            weight_per_label = defaultdict(float)
            for nb, w in nbrs.items():
                weight_per_label[labels[nb]] += w
            if not weight_per_label:
                continue
            max_w = max(weight_per_label.values())
            best = [lab for lab, w in weight_per_label.items() if w == max_w]
            new_lab = min(best)
            if new_lab != labels[q]:
                labels[q] = new_lab
                changed = True
        if not changed:
            break

    communities = defaultdict(list)
    for q, lab in labels.items():
        communities[lab].append(q)
    community_list = sorted(communities.values(), key=lambda c: -len(c))

    phys_adj = defaultdict(set)
    phys_nodes = set()
    for (a, b) in self.backend_connections:
        if a == b:
            continue
        phys_adj[a].add(b)
        phys_adj[b].add(a)
        phys_nodes.add(a)
        phys_nodes.add(b)
    phys_nodes = {p for p in phys_nodes if 0 <= p < N}
    phys_degree = {p: len(phys_adj[p]) for p in phys_nodes}

    def dist(p1, p2):
        if p1 == p2:
            return 0
        try:
            d = self.distance_matrix[p1][p2]
            if d and d > 0:
                return d
        except Exception:
            pass
        visited = {p1: 0}
        dq = deque([p1])
        while dq:
            x = dq.popleft()
            for y in phys_adj.get(x, ()):
                if y not in visited:
                    visited[y] = visited[x] + 1
                    if y == p2:
                        return visited[y]
                    dq.append(y)
        return 10 ** 6

    used_phys = set()
    logical_to_physical = {}

    for comm in community_list:
        size = len(comm)
        available = [p for p in phys_nodes if p not in used_phys]
        if not available:
            break
        available.sort(key=lambda p: (-phys_degree.get(p, 0), p))
        seed = available[0]

        region = []
        visited = {seed}
        frontier = deque([seed])
        while frontier and len(region) < size:
            node = frontier.popleft()
            if node in used_phys:
                continue
            region.append(node)
            nbs = [nb for nb in phys_adj[node]
                   if nb not in visited and nb not in used_phys and nb in phys_nodes]
            nbs.sort(key=lambda p: (-phys_degree.get(p, 0), p))
            for nb in nbs:
                visited.add(nb)
                frontier.append(nb)

        if len(region) < size:
            for p in available:
                if p not in region and p not in used_phys:
                    region.append(p)
                    if len(region) >= size:
                        break
        region = region[:size]
        if not region:
            continue

        comm_set = set(comm)
        intra_weight = {}
        for q in comm:
            w = 0.0
            for nb, ww in qig.get(q, {}).items():
                if nb in comm_set:
                    w += ww
            intra_weight[q] = w
        ordered_logical = sorted(comm, key=lambda q: (-intra_weight.get(q, 0.0), q))

        region_by_deg = sorted(region, key=lambda p: (-phys_degree.get(p, 0), p))
        placed = {}
        anchor_log = ordered_logical[0]
        anchor_phys = region_by_deg[0]
        placed[anchor_log] = anchor_phys

        for log_q in ordered_logical[1:]:
            partners = [(nb, w) for nb, w in qig.get(log_q, {}).items() if nb in placed]
            taken = set(placed.values())
            candidates = [p for p in region if p not in taken]
            if not candidates:
                break
            best_p = None
            best_cost = float('inf')
            for p in candidates:
                if partners:
                    cost = 0.0
                    for partner_log, w in partners:
                        cost += w * dist(p, placed[partner_log])
                else:
                    cost = -phys_degree.get(p, 0)
                if cost < best_cost:
                    best_cost = cost
                    best_p = p
            if best_p is None:
                best_p = candidates[0]
            placed[log_q] = best_p

        for log_q, phys_q in placed.items():
            if log_q not in logical_to_physical and phys_q not in used_phys:
                logical_to_physical[log_q] = phys_q
                used_phys.add(phys_q)

    final_mapping = [-1] * N
    final_used = [False] * N

    for log_q, phys_q in logical_to_physical.items():
        if 0 <= log_q < N and 0 <= phys_q < N and not final_used[phys_q] and final_mapping[log_q] == -1:
            final_mapping[log_q] = phys_q
            final_used[phys_q] = True

    for i in range(N):
        if final_mapping[i] == -1 and not final_used[i]:
            final_mapping[i] = i
            final_used[i] = True

    unused = [p for p in range(N) if not final_used[p]]
    idx = 0
    for i in range(N):
        if final_mapping[i] == -1:
            final_mapping[i] = unused[idx]
            final_used[unused[idx]] = True
            idx += 1

    self.mapping_dict = final_mapping
    self.reverse_mapping_dict = [0] * N
    for log_q in range(N):
        self.reverse_mapping_dict[self.mapping_dict[log_q]] = log_q

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)