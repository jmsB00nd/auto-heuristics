def init_mapping(self):
    from collections import defaultdict, deque

    N = self.num_qubits

    # ---------- 1. Build logical interaction weights ----------
    edge_weight = defaultdict(int)        # frozenset({a,b}) -> count
    neighbor_weight = defaultdict(lambda: defaultdict(int))  # a -> {b: count}
    logical_qubits = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                logical_qubits.add(a)
                continue
            key = frozenset((a, b))
            edge_weight[key] += 1
            neighbor_weight[a][b] += 1
            neighbor_weight[b][a] += 1
            logical_qubits.add(a)
            logical_qubits.add(b)
        else:
            for q in qubits:
                logical_qubits.add(q)

    # ---------- 2. Extract heavy weighted logical path ----------
    def heavy_path():
        if not edge_weight:
            return []
        # Start from the heaviest edge
        start_edge = max(edge_weight.items(), key=lambda kv: kv[1])[0]
        u, v = tuple(start_edge)
        path = [u, v]
        in_path = {u, v}

        def best_extension(node):
            best_n, best_w = None, -1
            for nb, w in neighbor_weight[node].items():
                if nb in in_path:
                    continue
                if w > best_w:
                    best_w = w
                    best_n = nb
            return best_n

        # Extend tail
        while True:
            nxt = best_extension(path[-1])
            if nxt is None:
                break
            path.append(nxt)
            in_path.add(nxt)
        # Extend head
        while True:
            nxt = best_extension(path[0])
            if nxt is None:
                break
            path.insert(0, nxt)
            in_path.add(nxt)
        return path

    h_path = heavy_path()

    # ---------- 3. Find a long simple path in the physical graph ----------
    def physical_degree(p):
        nbrs = self.backend[p] if p < len(self.backend) else []
        return len(nbrs)

    def long_physical_path(target_len):
        if target_len <= 0:
            return []
        # Try seeds in descending degree
        seeds = sorted(range(N), key=lambda p: -physical_degree(p))
        best = []
        # DFS with depth budget; cap exploration to keep it tractable
        for seed in seeds:
            stack = [(seed, [seed], {seed})]
            local_best = [seed]
            steps = 0
            STEP_CAP = 5000
            while stack and steps < STEP_CAP:
                node, path, visited = stack.pop()
                if len(path) > len(local_best):
                    local_best = path
                    if len(local_best) >= target_len:
                        break
                if len(path) >= target_len:
                    continue
                nbrs = self.backend[node] if node < len(self.backend) else []
                # explore higher-degree neighbors first
                nbr_sorted = sorted(
                    [n for n in nbrs if n not in visited],
                    key=lambda n: physical_degree(n),
                )
                for nb in nbr_sorted:
                    new_visited = visited | {nb}
                    stack.append((nb, path + [nb], new_visited))
                    steps += 1
                    if steps >= STEP_CAP:
                        break
            if len(local_best) > len(best):
                best = local_best
            if len(best) >= target_len:
                break
        return best

    phys_path = long_physical_path(len(h_path))

    # ---------- 4. Initialize mapping containers ----------
    mapping = [-1] * N
    reverse = [-1] * N
    placed_logical = set()
    used_physical = set()

    # Embed heavy path onto physical path
    for i in range(min(len(h_path), len(phys_path))):
        L = h_path[i]
        P = phys_path[i]
        if L >= N or P >= N:
            continue
        if L in placed_logical or P in used_physical:
            continue
        mapping[L] = P
        reverse[P] = L
        placed_logical.add(L)
        used_physical.add(P)

    # ---------- 5. Attach remaining logical qubits ----------
    def free_physical_neighbor_bfs(start_phys):
        # BFS on backend from start_phys to find the nearest free physical qubit
        visited = {start_phys}
        dq = deque([start_phys])
        while dq:
            p = dq.popleft()
            if p != start_phys and p not in used_physical:
                return p
            nbrs = self.backend[p] if p < len(self.backend) else []
            for nb in nbrs:
                if nb not in visited:
                    visited.add(nb)
                    dq.append(nb)
        return None

    # Process unplaced logical qubits in descending weighted-degree to placed set
    def weight_to_placed(L):
        return sum(w for nb, w in neighbor_weight[L].items() if nb in placed_logical)

    remaining_logical = [L for L in logical_qubits if L not in placed_logical and L < N]
    remaining_logical.sort(key=lambda L: -sum(neighbor_weight[L].values()))

    for L in list(remaining_logical):
        if L in placed_logical:
            continue
        # pick highest-frequency placed neighbor
        best_nb, best_w = None, -1
        for nb, w in neighbor_weight[L].items():
            if nb in placed_logical and w > best_w:
                best_w = w
                best_nb = nb
        target_phys = None
        if best_nb is not None:
            anchor_phys = mapping[best_nb]
            if anchor_phys >= 0:
                target_phys = free_physical_neighbor_bfs(anchor_phys)
        if target_phys is None:
            # nearest free physical via distance matrix from any placed anchor of L's neighborhood
            for nb in sorted(neighbor_weight[L].keys(),
                             key=lambda n: -neighbor_weight[L][n]):
                if nb in placed_logical:
                    ap = mapping[nb]
                    cand = None
                    best_d = float("inf")
                    for p in range(N):
                        if p in used_physical:
                            continue
                        d = self.distance_matrix[ap][p]
                        if d < best_d:
                            best_d = d
                            cand = p
                    target_phys = cand
                    break
        if target_phys is None:
            # any free physical qubit
            for p in range(N):
                if p not in used_physical:
                    target_phys = p
                    break
        if target_phys is None:
            continue
        mapping[L] = target_phys
        reverse[target_phys] = L
        placed_logical.add(L)
        used_physical.add(target_phys)

    # ---------- 6. Identity-style fill for any remaining logical ids ----------
    free_phys_iter = (p for p in range(N) if p not in used_physical)
    for L in range(N):
        if mapping[L] != -1:
            continue
        # Prefer identity if free
        if L < N and L not in used_physical:
            mapping[L] = L
            reverse[L] = L
            used_physical.add(L)
            continue
        try:
            p = next(free_phys_iter)
        except StopIteration:
            p = None
        if p is None:
            continue
        mapping[L] = p
        reverse[p] = L
        used_physical.add(p)

    # Final safety: if any slot still -1 (shouldn't happen), assign identity
    for L in range(N):
        if mapping[L] == -1:
            for p in range(N):
                if p not in used_physical:
                    mapping[L] = p
                    reverse[p] = L
                    used_physical.add(p)
                    break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)