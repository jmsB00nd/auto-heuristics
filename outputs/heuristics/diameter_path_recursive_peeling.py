def init_mapping(self):
    import collections
    import heapq

    N = self.num_qubits

    logical_qubits = set()
    edge_weight = collections.defaultdict(float)
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig:
        for u, nbrs in qig.items():
            for v, w in nbrs.items():
                if u == v:
                    continue
                logical_qubits.add(u); logical_qubits.add(v)
                a, b = (u, v) if u < v else (v, u)
                edge_weight[(a, b)] = max(edge_weight[(a, b)], float(w))
    if not edge_weight:
        for _gid, qs in self.access.items():
            if len(qs) == 2:
                u, v = qs[0], qs[1]
                if u == v:
                    continue
                logical_qubits.add(u); logical_qubits.add(v)
                a, b = (u, v) if u < v else (v, u)
                edge_weight[(a, b)] += 1.0
    for _gid, qs in self.access.items():
        for q in qs:
            logical_qubits.add(q)

    log_adj = collections.defaultdict(dict)
    for (a, b), w in edge_weight.items():
        log_adj[a][b] = w
        log_adj[b][a] = w

    def heaviest_path_in_component(nodes_set):
        if not nodes_set:
            return []
        sub_adj = {n: {m: w for m, w in log_adj[n].items() if m in nodes_set} for n in nodes_set}

        def far_from(src):
            best_node, best_cost, best_path = src, 0.0, [src]
            visited = {src: (0.0, [src])}
            pq = [(-0.0, src)]
            while pq:
                neg_c, u = heapq.heappop(pq)
                c = -neg_c
                if c < visited[u][0] - 1e-12:
                    continue
                path_u = visited[u][1]
                if c > best_cost:
                    best_cost, best_node, best_path = c, u, path_u
                for v, w in sub_adj[u].items():
                    if v in path_u:
                        continue
                    nc = c + w
                    if v not in visited or nc > visited[v][0]:
                        new_path = path_u + [v]
                        visited[v] = (nc, new_path)
                        heapq.heappush(pq, (-nc, v))
            return best_node, best_path

        start = max(nodes_set, key=lambda n: sum(sub_adj[n].values()) if sub_adj[n] else 0.0)
        end_a, _ = far_from(start)
        end_b, path = far_from(end_a)
        return path

    def physical_longest_path(avail_phys):
        if not avail_phys:
            return []
        avail = set(avail_phys)
        adj = {p: [q for q in self.backend[p] if q in avail] for p in avail}

        def bfs_far(src):
            parent = {src: None}
            order = [src]
            head = 0
            while head < len(order):
                u = order[head]; head += 1
                for v in adj[u]:
                    if v not in parent:
                        parent[v] = u
                        order.append(v)
            far = max(order, key=lambda x: self.distance_matrix[src][x] if self.distance_matrix[src][x] > 0 or x == src else -1)
            path = []
            cur = far
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()
            return far, path

        seed = next(iter(avail))
        a, _ = bfs_far(seed)
        b, path = bfs_far(a)
        return path

    def components(nodes_set):
        seen = set()
        comps = []
        for n in nodes_set:
            if n in seen:
                continue
            stack = [n]; comp = []
            seen.add(n)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in log_adj[u]:
                    if v in nodes_set and v not in seen:
                        seen.add(v); stack.append(v)
            comps.append(comp)
        comps.sort(key=lambda c: -sum(log_adj[u].get(v, 0.0) for u in c for v in c if v in set(c)))
        return comps

    mapping = [-1] * N
    reverse = [-1] * N
    used_physical = set()
    placed_logical = set()

    remaining_logical = set(logical_qubits)

    def place_pair(lq, pq):
        mapping[lq] = pq
        reverse[pq] = lq
        used_physical.add(pq)
        placed_logical.add(lq)

    def recurse(node_set):
        if not node_set:
            return
        for comp in components(node_set):
            comp_set = set(comp)
            spine = heaviest_path_in_component(comp_set)
            if not spine:
                continue
            avail_phys = [p for p in range(N) if p not in used_physical]
            phys_spine = physical_longest_path(avail_phys)
            if not phys_spine:
                return
            k = min(len(spine), len(phys_spine))
            for i in range(k):
                place_pair(spine[i], phys_spine[i])
            leftover = comp_set - placed_logical
            if leftover:
                recurse(leftover)

    recurse(remaining_logical)

    leftover_logicals = [q for q in logical_qubits if q not in placed_logical]
    centrality = getattr(self, "physical_centrality", {}) or {}
    leftover_logicals.sort(key=lambda q: -float(self.logical_activity.get(q, 0)) if hasattr(self, "logical_activity") else 0)
    for lq in leftover_logicals:
        avail = [p for p in range(N) if p not in used_physical]
        if not avail:
            break
        avail.sort(key=lambda p: -centrality.get(p, 0.0))
        place_pair(lq, avail[0])

    for lq in range(N):
        if mapping[lq] == -1:
            for pq in range(N):
                if pq not in used_physical:
                    place_pair(lq, pq)
                    break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)