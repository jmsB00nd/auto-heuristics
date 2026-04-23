def init_mapping(self):
    import heapq
    from collections import deque

    N = self.num_qubits
    mapping = [-1] * N
    reverse = [-1] * N
    used = set()
    placed = set()

    qig = self.qubit_interaction_graph
    backend = self.backend
    centrality = self.physical_centrality
    phys_range = len(self.distance_matrix)

    def place(lq, pq):
        mapping[lq] = pq
        reverse[pq] = lq
        used.add(pq)
        placed.add(lq)

    def pick_central_unused():
        best, best_score = None, -1.0
        for p in range(phys_range):
            if p in used:
                continue
            sc = centrality.get(p, 0.0)
            if sc > best_score:
                best_score = sc
                best = p
        if best is not None:
            return best
        for p in range(N):
            if p not in used:
                return p
        return None

    def pick_adjacent_unused(anchor):
        best, best_score = None, -1.0
        for nb in backend.get(anchor, []):
            if nb in used:
                continue
            sc = centrality.get(nb, 0.0)
            if sc > best_score:
                best_score = sc
                best = nb
        if best is not None:
            return best
        visited = {anchor}
        bq = deque([anchor])
        while bq:
            cur = bq.popleft()
            for nb in backend.get(cur, []):
                if nb in visited:
                    continue
                visited.add(nb)
                if nb not in used:
                    return nb
                bq.append(nb)
        return None

    edges = []
    for q1 in qig:
        for q2, w in qig[q1].items():
            if q1 < q2:
                edges.append((w, q1, q2))
    edges.sort(key=lambda x: -x[0])

    heap = []

    def add_frontier(lq):
        for nb, w in qig.get(lq, {}).items():
            if nb not in placed:
                heapq.heappush(heap, (-w, lq, nb))

    def grow():
        while heap:
            _, pq, uq = heapq.heappop(heap)
            if uq in placed:
                continue
            new_phys = pick_adjacent_unused(mapping[pq])
            if new_phys is None:
                return
            place(uq, new_phys)
            add_frontier(uq)

    for (w, q1, q2) in edges:
        if q1 in placed or q2 in placed:
            continue
        p1 = pick_central_unused()
        if p1 is None:
            break
        used.add(p1)
        p2 = pick_adjacent_unused(p1)
        used.discard(p1)
        place(q1, p1)
        add_frontier(q1)
        if p2 is not None:
            place(q2, p2)
            add_frontier(q2)
        grow()

    for q in qig:
        if 0 <= q < N and q not in placed:
            p = pick_central_unused()
            if p is None:
                break
            place(q, p)
            add_frontier(q)
            grow()

    for q in range(N):
        if mapping[q] == -1 and q not in used:
            place(q, q)
    for q in range(N):
        if mapping[q] != -1:
            continue
        for p in range(N):
            if p not in used:
                place(q, p)
                break

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)