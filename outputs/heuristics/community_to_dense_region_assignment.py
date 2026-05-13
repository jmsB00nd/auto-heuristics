def init_mapping(self):
    import collections
    import heapq
    import random

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    # ---- 1. Build weighted logical interaction graph from self.access ----
    logical_qubits = set()
    log_adj = collections.defaultdict(lambda: collections.defaultdict(int))
    for _gid, qubits in self.access.items():
        for q in qubits:
            logical_qubits.add(q)
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a != b:
                log_adj[a][b] += 1
                log_adj[b][a] += 1

    # ---- 2. Community detection via weighted label propagation ----
    nodes = sorted(logical_qubits)
    labels = {q: q for q in nodes}
    if nodes:
        rng = random.Random(1234)
        for _ in range(25):
            order = list(nodes)
            rng.shuffle(order)
            changed = False
            for q in order:
                neigh = log_adj[q]
                if not neigh:
                    continue
                score = collections.defaultdict(float)
                for nb, w in neigh.items():
                    score[labels[nb]] += w
                best = max(score.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                if labels[q] != best:
                    labels[q] = best
                    changed = True
            if not changed:
                break

    communities = collections.defaultdict(list)
    for q in nodes:
        communities[labels[q]].append(q)

    def _comm_weight(c):
        return sum(self.logical_activity[q] for q in c)

    comm_list = sorted(communities.values(),
                       key=lambda c: (-_comm_weight(c), -len(c)))

    # ---- 3. Carve compact physical regions via BFS from central seeds ----
    used_phys = set()
    phys_order = sorted(range(N),
                        key=lambda p: -self.physical_centrality.get(p, 0.0))

    for comm in comm_list:
        if len(used_phys) >= N or not comm:
            continue
        size = min(len(comm), N - len(used_phys))
        comm = comm[:size]

        seed = next((p for p in phys_order if p not in used_phys), None)
        if seed is None:
            break

        region = []
        visited_local = {seed}
        frontier = [(-self.physical_centrality.get(seed, 0.0), seed)]
        while frontier and len(region) < size:
            _, p = heapq.heappop(frontier)
            if p in used_phys:
                continue
            region.append(p)
            for nb in self.backend.get(p, ()):
                if nb not in visited_local and nb not in used_phys:
                    visited_local.add(nb)
                    heapq.heappush(frontier,
                                   (-self.physical_centrality.get(nb, 0.0), nb))

        if len(region) < size:
            anchor = region[0] if region else seed
            need = size - len(region)
            in_region = set(region)
            extras = [p for p in range(N)
                      if p not in used_phys and p not in in_region]
            extras.sort(key=lambda p: (self.distance_matrix[anchor][p],
                                       -self.physical_centrality.get(p, 0.0)))
            region.extend(extras[:need])

        # ---- 4. Intra-region greedy placement ----
        comm_sorted = sorted(comm, key=lambda q: -self.logical_activity[q])
        region_sorted = sorted(region,
                               key=lambda p: -self.physical_centrality.get(p, 0.0))

        placed = {}
        avail = list(region_sorted)

        if comm_sorted and avail:
            placed[comm_sorted[0]] = avail.pop(0)

        for l in comm_sorted[1:]:
            if not avail:
                break
            best_p = avail[0]
            best_key = None
            for p in avail:
                s = 0.0
                for pl, pp in placed.items():
                    w = log_adj[l].get(pl, 0)
                    if w:
                        s -= w * self.distance_matrix[p][pp]
                key = (s, self.physical_centrality.get(p, 0.0))
                if best_key is None or key > best_key:
                    best_key = key
                    best_p = p
            placed[l] = best_p
            avail.remove(best_p)

        for l, p in placed.items():
            self.mapping_dict[l] = p
            self.reverse_mapping_dict[p] = l
            used_phys.add(p)

    # ---- 5. Backfill unassigned logicals onto remaining physicals ----
    remaining_phys = [p for p in range(N) if p not in used_phys]
    rp_idx = 0
    for l in range(N):
        if self.mapping_dict[l] == -1:
            p = remaining_phys[rp_idx]
            rp_idx += 1
            self.mapping_dict[l] = p
            self.reverse_mapping_dict[p] = l

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)