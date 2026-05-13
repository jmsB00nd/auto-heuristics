def init_mapping(self):
    import heapq
    import math
    from collections import defaultdict

    N = self.num_qubits
    self.mapping_dict = [-1] * N
    self.reverse_mapping_dict = [-1] * N

    interactions = []
    activity = defaultdict(int)
    logical_set = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            interactions.append((a, b))
            activity[a] += 1
            activity[b] += 1
            logical_set.add(a)
            logical_set.add(b)
        elif len(qubits) == 1:
            logical_set.add(qubits[0])

    neighbor_weight = defaultdict(lambda: defaultdict(int))
    for a, b in interactions:
        neighbor_weight[a][b] += 1
        neighbor_weight[b][a] += 1

    def dijkstra(src):
        dist = [math.inf] * N
        prev = [-1] * N
        dist[src] = 0
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v in self.backend[u] if u < len(self.backend) else []:
                nd = d + 1
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))
        return dist, prev

    def path_to(prev, target):
        path = []
        cur = target
        while cur != -1:
            path.append(cur)
            cur = prev[cur]
        path.reverse()
        return path

    centrality = []
    for p in range(N):
        row = self.distance_matrix[p]
        s = 0
        for q in range(N):
            d = row[q]
            if d == 0 and p != q:
                s += N
            else:
                s += d
        centrality.append((s, p))
    centrality.sort()

    active_logicals = sorted(logical_set, key=lambda x: -activity[x])
    K = min(len(active_logicals), max(1, int(math.isqrt(max(N, 1)))))
    terminals = active_logicals[:K]
    centers = [p for _, p in centrality[:K]]

    steiner_nodes = set()
    if centers:
        steiner_nodes.add(centers[0])
        remaining = set(centers[1:])
        while remaining:
            best_node = None
            best_path = None
            best_len = math.inf
            for r in list(remaining):
                dist, prev = dijkstra(r)
                local_best = math.inf
                local_target = -1
                for s in steiner_nodes:
                    if dist[s] < local_best:
                        local_best = dist[s]
                        local_target = s
                if local_best < best_len:
                    best_len = local_best
                    best_node = r
                    best_path = path_to(prev, local_target)
            if best_node is None or best_path is None:
                break
            for node in best_path:
                steiner_nodes.add(node)
            steiner_nodes.add(best_node)
            remaining.discard(best_node)

    used = set()

    terminal_assignment = {}
    available_centers = list(centers)
    for term in terminals:
        if not available_centers:
            break
        best_c = None
        best_cost = math.inf
        for c in available_centers:
            if c in used:
                continue
            placed_neighbors = [(nb, w) for nb, w in neighbor_weight[term].items()
                                if nb in terminal_assignment]
            if placed_neighbors:
                cost = sum(w * self.distance_matrix[c][terminal_assignment[nb]]
                           for nb, w in placed_neighbors)
            else:
                cost = 0
            if cost < best_cost:
                best_cost = cost
                best_c = c
        if best_c is None:
            break
        terminal_assignment[term] = best_c
        used.add(best_c)
        available_centers.remove(best_c)

    for term, phys in terminal_assignment.items():
        if term < N and self.mapping_dict[term] == -1:
            self.mapping_dict[term] = phys
            self.reverse_mapping_dict[phys] = term

    preferred_pool = [n for n in steiner_nodes if n not in used]
    preferred_pool.sort(key=lambda p: centrality[p][0] if False else
                       sum(self.distance_matrix[p]))

    placed_logicals = set(t for t in terminal_assignment if t < N
                          and self.mapping_dict[t] != -1)
    remaining_logicals = [l for l in active_logicals if l not in placed_logicals and l < N]

    def pick_physical(logical, candidate_pool):
        placed_neighbors = [(nb, w) for nb, w in neighbor_weight[logical].items()
                            if nb in placed_logicals and nb < N
                            and self.mapping_dict[nb] != -1]
        best_p = None
        best_cost = math.inf
        for p in candidate_pool:
            if p in used or p >= N:
                continue
            if placed_neighbors:
                cost = sum(w * self.distance_matrix[p][self.mapping_dict[nb]]
                           for nb, w in placed_neighbors)
            else:
                cost = sum(self.distance_matrix[p])
            if cost < best_cost:
                best_cost = cost
                best_p = p
        return best_p

    for logical in remaining_logicals:
        chosen = pick_physical(logical, preferred_pool)
        if chosen is None:
            full_pool = [p for p in range(N) if p not in used]
            chosen = pick_physical(logical, full_pool)
        if chosen is None:
            continue
        self.mapping_dict[logical] = chosen
        self.reverse_mapping_dict[chosen] = logical
        used.add(chosen)
        placed_logicals.add(logical)

    unused_physical = [p for p in range(N) if p not in used]
    up_idx = 0
    for logical in range(N):
        if self.mapping_dict[logical] == -1:
            while up_idx < len(unused_physical) and unused_physical[up_idx] in used:
                up_idx += 1
            if up_idx >= len(unused_physical):
                break
            phys = unused_physical[up_idx]
            up_idx += 1
            self.mapping_dict[logical] = phys
            self.reverse_mapping_dict[phys] = logical
            used.add(phys)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)