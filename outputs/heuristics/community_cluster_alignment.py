def init_mapping(self):
    import collections
    import heapq

    N = self.num_qubits
    self.mapping_dict = [0] * N
    self.reverse_mapping_dict = [0] * N

    # ---- 1. Build weighted logical interaction graph from self.access ----
    edge_w = collections.defaultdict(int)
    log_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            u, v = (a, b) if a < b else (b, a)
            edge_w[(u, v)] += 1
            log_nodes.add(a)
            log_nodes.add(b)

    log_adj = collections.defaultdict(dict)
    for (u, v), w in edge_w.items():
        log_adj[u][v] = w
        log_adj[v][u] = w

    log_weighted_degree = {q: sum(log_adj[q].values()) for q in log_nodes}

    # ---- 2a. Community detection on logical graph (label propagation) ----
    label = {q: q for q in log_nodes}
    nodes_list = list(log_nodes)
    if nodes_list:
        rng_order = sorted(nodes_list, key=lambda x: -log_weighted_degree.get(x, 0))
        for _ in range(8):
            changed = False
            for q in rng_order:
                if not log_adj[q]:
                    continue
                cnt = collections.defaultdict(float)
                for nb, w in log_adj[q].items():
                    cnt[label[nb]] += w
                best = max(cnt.items(), key=lambda kv: (kv[1], -kv[0]))[0]
                if label[q] != best:
                    label[q] = best
                    changed = True
            if not changed:
                break

    communities = collections.defaultdict(list)
    for q, lab in label.items():
        communities[lab].append(q)
    comm_list = list(communities.values())

    def comm_internal_weight(c):
        s = set(c)
        tot = 0
        for q in c:
            for nb, w in log_adj[q].items():
                if nb in s:
                    tot += w
        return tot // 2

    comm_list.sort(key=lambda c: (-len(c), -comm_internal_weight(c)))

    # ---- 2b. Hardware regions: BFS-grown dense balls ----
    phys_deg = [len(self.backend[p]) if p < len(self.backend) else 0 for p in range(N)]
    used_phys = set()
    region_seeds_order = sorted(range(N), key=lambda p: -phys_deg[p])

    regions = []
    assigned_phys = set()
    for comm in comm_list:
        size_needed = len(comm)
        if size_needed == 0:
            continue
        seed = None
        for p in region_seeds_order:
            if p not in assigned_phys:
                seed = p
                break
        if seed is None:
            regions.append([])
            continue
        # BFS grow region from seed picking closest free neighbors
        region = []
        visited = {seed}
        heap = [(0, seed)]
        while heap and len(region) < size_needed:
            d, p = heapq.heappop(heap)
            if p in assigned_phys:
                continue
            region.append(p)
            assigned_phys.add(p)
            for nb in self.backend[p] if p < len(self.backend) else []:
                if nb not in visited and nb not in assigned_phys:
                    visited.add(nb)
                    dist = self.distance_matrix[seed][nb] if seed < len(self.distance_matrix) and nb < len(self.distance_matrix[seed]) else d + 1
                    heapq.heappush(heap, (dist, nb))
        # If region not big enough, fill with closest unused physicals globally
        if len(region) < size_needed:
            remaining = [p for p in range(N) if p not in assigned_phys]
            remaining.sort(key=lambda p: self.distance_matrix[seed][p] if seed < len(self.distance_matrix) and p < len(self.distance_matrix[seed]) else N)
            for p in remaining:
                if len(region) >= size_needed:
                    break
                region.append(p)
                assigned_phys.add(p)
        regions.append(region)

    # ---- 3 & 4. Place each community into its region ----
    placed_log = set()
    used_phys = set()

    for comm, region in zip(comm_list, regions):
        if not comm or not region:
            continue
        comm_sorted = sorted(comm, key=lambda q: -log_weighted_degree.get(q, 0))
        region_set = set(region)
        # Seed: highest-degree logical -> highest-degree physical in region
        first_log = comm_sorted[0]
        region_by_deg = sorted(region, key=lambda p: -phys_deg[p])
        first_phys = None
        for p in region_by_deg:
            if p not in used_phys:
                first_phys = p
                break
        if first_phys is None:
            continue
        self.mapping_dict[first_log] = first_phys
        placed_log.add(first_log)
        used_phys.add(first_phys)

        remaining_log = [q for q in comm_sorted[1:]]
        while remaining_log:
            best_q = None
            best_score = -1
            for q in remaining_log:
                score = sum(w for nb, w in log_adj[q].items() if nb in placed_log)
                if score > best_score:
                    best_score = score
                    best_q = q
            if best_q is None:
                best_q = remaining_log[0]
            placed_neighbors_phys = [self.mapping_dict[nb] for nb in log_adj[best_q] if nb in placed_log]
            free_in_region = [p for p in region if p not in used_phys]
            target_phys = None
            if free_in_region:
                if placed_neighbors_phys:
                    def cost(p):
                        return sum(self.distance_matrix[p][pn] if p < len(self.distance_matrix) and pn < len(self.distance_matrix[p]) else N for pn in placed_neighbors_phys)
                    target_phys = min(free_in_region, key=cost)
                else:
                    target_phys = max(free_in_region, key=lambda p: phys_deg[p])
            else:
                free_global = [p for p in range(N) if p not in used_phys]
                if free_global:
                    if placed_neighbors_phys:
                        def cost2(p):
                            return sum(self.distance_matrix[p][pn] if p < len(self.distance_matrix) and pn < len(self.distance_matrix[p]) else N for pn in placed_neighbors_phys)
                        target_phys = min(free_global, key=cost2)
                    else:
                        target_phys = free_global[0]
            if target_phys is None:
                break
            self.mapping_dict[best_q] = target_phys
            placed_log.add(best_q)
            used_phys.add(target_phys)
            remaining_log.remove(best_q)

    # ---- 5. Identity-fallback for any remaining logical ids ----
    free_phys_pool = [p for p in range(N) if p not in used_phys]
    fp_idx = 0
    for L in range(N):
        if L in placed_log:
            continue
        # Prefer identity if available
        if L not in used_phys:
            self.mapping_dict[L] = L
            used_phys.add(L)
            placed_log.add(L)
        else:
            while fp_idx < len(free_phys_pool) and free_phys_pool[fp_idx] in used_phys:
                fp_idx += 1
            if fp_idx < len(free_phys_pool):
                p = free_phys_pool[fp_idx]
                self.mapping_dict[L] = p
                used_phys.add(p)
                placed_log.add(L)
                fp_idx += 1

    # ---- 6. Build reverse mapping ----
    for L in range(N):
        P = self.mapping_dict[L]
        if 0 <= P < N:
            self.reverse_mapping_dict[P] = L

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)