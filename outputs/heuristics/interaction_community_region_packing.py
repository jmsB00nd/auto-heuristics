def init_mapping(self):
    import random
    from collections import defaultdict, deque

    N = self.num_qubits

    logical_set = set()
    interactions = []
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            logical_set.add(a)
            logical_set.add(b)
            interactions.append((a, b))

    qig = self.qubit_interaction_graph
    adj_w = defaultdict(lambda: defaultdict(float))
    for u in logical_set:
        if u in qig:
            for v, w in qig[u].items():
                if v in logical_set and v != u and w > 0:
                    adj_w[u][v] = float(w)
    for a, b in interactions:
        if b not in adj_w[a]:
            adj_w[a][b] = adj_w[a].get(b, 0.0) + 1.0
            adj_w[b][a] = adj_w[b].get(a, 0.0) + 1.0

    rng = random.Random(0xC0FFEE)
    labels = {q: q for q in logical_set}
    nodes = sorted(logical_set)
    for _ in range(8):
        order = nodes[:]
        rng.shuffle(order)
        changed = False
        for u in order:
            if not adj_w[u]:
                continue
            score = defaultdict(float)
            for v, w in adj_w[u].items():
                score[labels[v]] += w
            best_w = -1.0
            best_labels = []
            for lab, sw in score.items():
                if sw > best_w + 1e-12:
                    best_w = sw
                    best_labels = [lab]
                elif abs(sw - best_w) <= 1e-12:
                    best_labels.append(lab)
            new_label = labels[u] if labels[u] in best_labels else rng.choice(best_labels)
            if new_label != labels[u]:
                labels[u] = new_label
                changed = True
        if not changed:
            break

    communities = defaultdict(list)
    for q, lab in labels.items():
        communities[lab].append(q)
    community_list = sorted(communities.values(),
                            key=lambda c: (-len(c),
                                           -sum(self.logical_activity.get(q, 0) for q in c)))

    centrality = self.physical_centrality
    backend = self.backend
    dist = self.distance_matrix

    sorted_physicals_by_centrality = sorted(range(N),
                                            key=lambda p: (-centrality.get(p, 0.0), p))

    mapping = [-1] * N
    reverse = [-1] * N
    used_physical = set()
    placed_logical = set()

    def grow_region(seed, size, forbidden):
        region = []
        visited = {seed}
        q = deque([seed])
        while q and len(region) < size:
            cur = q.popleft()
            if cur in forbidden:
                continue
            region.append(cur)
            neighbors = sorted(backend.get(cur, ()),
                               key=lambda p: (-centrality.get(p, 0.0), p))
            for nb in neighbors:
                if nb not in visited and nb not in forbidden:
                    visited.add(nb)
                    q.append(nb)
        return region if len(region) == size else None

    for community in community_list:
        size = len(community)
        if size == 0:
            continue
        seed = None
        for p in sorted_physicals_by_centrality:
            if p not in used_physical:
                seed = p
                break
        if seed is None:
            break
        region = grow_region(seed, size, used_physical)
        if region is None:
            region = [p for p in sorted_physicals_by_centrality
                      if p not in used_physical][:size]
            if len(region) < size:
                continue

        community_sorted = sorted(community,
                                  key=lambda q: (-self.logical_activity.get(q, 0), q))
        region_sorted = sorted(region,
                               key=lambda p: (-centrality.get(p, 0.0), p))

        first_logical = community_sorted[0]
        first_physical = region_sorted[0]
        mapping[first_logical] = first_physical
        reverse[first_physical] = first_logical
        used_physical.add(first_physical)
        placed_logical.add(first_logical)
        placed_in_region = [(first_logical, first_physical)]

        for lq in community_sorted[1:]:
            best_p = None
            best_cost = float('inf')
            for p in region:
                if p in used_physical:
                    continue
                cost = 0.0
                for (plq, ppq) in placed_in_region:
                    w = adj_w[lq].get(plq, 0.0)
                    if w > 0:
                        cost += w * dist[p][ppq]
                    else:
                        cost += 1e-6 * dist[p][ppq]
                if cost < best_cost:
                    best_cost = cost
                    best_p = p
            if best_p is None:
                continue
            mapping[lq] = best_p
            reverse[best_p] = lq
            used_physical.add(best_p)
            placed_logical.add(lq)
            placed_in_region.append((lq, best_p))

    unplaced_logicals = [q for q in sorted(logical_set,
                                           key=lambda x: (-self.logical_activity.get(x, 0), x))
                         if q not in placed_logical and q < N]
    free_physicals = [p for p in sorted_physicals_by_centrality if p not in used_physical]
    for lq in unplaced_logicals:
        if not free_physicals:
            break
        p = free_physicals.pop(0)
        mapping[lq] = p
        reverse[p] = lq
        used_physical.add(p)
        placed_logical.add(lq)

    remaining_logicals = [q for q in range(N) if mapping[q] == -1]
    remaining_physicals = [p for p in range(N) if p not in used_physical]
    for lq, p in zip(remaining_logicals, remaining_physicals):
        mapping[lq] = p
        reverse[p] = lq

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)