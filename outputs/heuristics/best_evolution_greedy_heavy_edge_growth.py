def init_mapping(self):
    import heapq
    from collections import defaultdict

    N = self.num_qubits

    # 1. Gather logical interaction edges and active logical qubits.
    edge_weight = defaultdict(int)
    active_logicals = set()
    qig = getattr(self, "qubit_interaction_graph", None)
    if qig:
        for u, nbrs in qig.items():
            if u >= N:
                continue
            for v, w in nbrs.items():
                if v >= N or u == v or w <= 0:
                    continue
                a, b = (u, v) if u < v else (v, u)
                # qig is symmetric; record once
                if (a, b) not in edge_weight:
                    edge_weight[(a, b)] = w
                active_logicals.add(u)
                active_logicals.add(v)
    if not edge_weight:
        for _, qubits in self.access.items():
            if len(qubits) == 2:
                u, v = qubits[0], qubits[1]
                if u == v or u >= N or v >= N:
                    continue
                a, b = (u, v) if u < v else (v, u)
                edge_weight[(a, b)] += 1
                active_logicals.add(u)
                active_logicals.add(v)

    # Per-logical weight to a set of already-placed logicals.
    def weight_to(logical, placed_set):
        s = 0
        if qig and logical in qig:
            for p in placed_set:
                s += qig[logical].get(p, 0)
            return s
        for (a, b), w in edge_weight.items():
            if a == logical and b in placed_set:
                s += w
            elif b == logical and a in placed_set:
                s += w
        return s

    placed = {}        # logical -> physical
    used = set()       # physical qubits taken

    centrality = getattr(self, "physical_centrality", {}) or {}
    backend = self.backend
    dmat = self.distance_matrix

    def cscore(p):
        return centrality.get(p, 0.0)

    # 2. Find most central physical edge.
    best_edge = None
    best_score = None
    seen_pe = set()
    for (pa, pb) in self.backend_connections:
        if pa == pb:
            continue
        key = (pa, pb) if pa < pb else (pb, pa)
        if key in seen_pe:
            continue
        seen_pe.add(key)
        sc = cscore(pa) + cscore(pb)
        if best_score is None or sc > best_score:
            best_score = sc
            best_edge = key

    # 3. Place heaviest logical edge onto most central physical edge.
    if edge_weight and best_edge is not None:
        heaviest = max(edge_weight.items(), key=lambda kv: kv[1])[0]
        la, lb = heaviest
        pa, pb = best_edge
        # Anchor the higher-activity endpoint on the more central physical.
        act = getattr(self, "logical_activity", {}) or {}
        if act.get(la, 0) < act.get(lb, 0):
            la, lb = lb, la
        if cscore(pa) < cscore(pb):
            pa, pb = pb, pa
        placed[la] = pa
        placed[lb] = pb
        used.add(pa)
        used.add(pb)

    # 4. Iteratively grow the placement.
    remaining = [q for q in active_logicals if q not in placed]
    while remaining:
        # Pick unplaced logical with largest interaction weight to placed set.
        best_l = None
        best_w = -1
        for l in remaining:
            w = weight_to(l, placed.keys())
            if w > best_w:
                best_w = w
                best_l = l
        if best_l is None:
            break

        # Find closest free physical neighbor of any placed logical
        # that interacts with best_l (prefer strongest interaction first).
        anchors = []
        for pl in placed:
            w = 0
            if qig and best_l in qig:
                w = qig[best_l].get(pl, 0)
            else:
                a, b = (best_l, pl) if best_l < pl else (pl, best_l)
                w = edge_weight.get((a, b), 0)
            if w > 0:
                anchors.append((-w, placed[pl]))
        anchors.sort()

        chosen_phys = None
        # BFS-style: search neighbors at increasing distance from anchors.
        if anchors:
            visited = set()
            heap = []
            for i, (_, ap) in enumerate(anchors):
                for nb in backend.get(ap, ()):  # 1-hop free first
                    if nb in used or nb in visited:
                        continue
                    heapq.heappush(heap, (1, i, nb))
                    visited.add(nb)
            # If no 1-hop free, expand to 2-hop, etc.
            while heap and chosen_phys is None:
                d, i, p = heapq.heappop(heap)
                if p in used:
                    continue
                chosen_phys = p
                break

            if chosen_phys is None:
                # Search outward from anchors using distance matrix.
                best_d = None
                for _, ap in anchors:
                    if ap >= len(dmat):
                        continue
                    for cand in range(N):
                        if cand in used:
                            continue
                        d = dmat[ap][cand] if cand < len(dmat[ap]) else None
                        if d is None or d <= 0:
                            continue
                        if best_d is None or d < best_d:
                            best_d = d
                            chosen_phys = cand

        # 5. Fallback: most central free physical qubit.
        if chosen_phys is None:
            free = [p for p in range(N) if p not in used]
            if not free:
                break
            chosen_phys = max(free, key=cscore)

        placed[best_l] = chosen_phys
        used.add(chosen_phys)
        remaining.remove(best_l)

    # 6. Build output lists; back-fill idle logicals onto free physicals by centrality.
    mapping = [-1] * N
    reverse = [-1] * N
    for l, p in placed.items():
        if 0 <= l < N and 0 <= p < N:
            mapping[l] = p
            reverse[p] = l

    free_phys = sorted([p for p in range(N) if p not in used],
                       key=lambda p: -cscore(p))
    idle_logicals = [l for l in range(N) if mapping[l] == -1]

    fi = 0
    for l in idle_logicals:
        if fi >= len(free_phys):
            break
        p = free_phys[fi]
        mapping[l] = p
        reverse[p] = l
        used.add(p)
        fi += 1

    # Final identity-style safety fill (in case anything is still unset).
    if any(m == -1 for m in mapping):
        taken = set(p for p in mapping if p != -1)
        leftover = [p for p in range(N) if p not in taken]
        li = 0
        for l in range(N):
            if mapping[l] == -1:
                if li < len(leftover):
                    p = leftover[li]
                    mapping[l] = p
                    reverse[p] = l
                    li += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse
    assert len(set(self.mapping_dict)) == len(self.mapping_dict)