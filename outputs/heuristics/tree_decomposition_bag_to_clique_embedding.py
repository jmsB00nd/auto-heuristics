def init_mapping(self):
    import heapq
    from collections import defaultdict, deque

    N = self.num_qubits

    # ---- 1. Extract logical interactions from self.access ----
    logical_nodes = set()
    edge_weight = defaultdict(int)
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            logical_nodes.add(a)
            logical_nodes.add(b)
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edge_weight[key] += 1
        elif len(qubits) == 1:
            logical_nodes.add(qubits[0])

    # Working adjacency for elimination
    adj = defaultdict(set)
    for (a, b) in edge_weight:
        adj[a].add(b)
        adj[b].add(a)
    for v in logical_nodes:
        _ = adj[v]  # ensure key exists

    # ---- 2. Min-fill-in elimination ordering ----
    def fill_in_count(v, adjacency):
        nbrs = list(adjacency[v])
        fill = 0
        for i in range(len(nbrs)):
            ni = nbrs[i]
            for j in range(i + 1, len(nbrs)):
                nj = nbrs[j]
                if nj not in adjacency[ni]:
                    fill += 1
        return fill

    remaining = set(logical_nodes)
    elim_order = []
    bags = []  # list of (eliminated_vertex, bag_set)
    work_adj = defaultdict(set)
    for v, ns in adj.items():
        work_adj[v] = set(ns)

    while remaining:
        best_v = None
        best_score = None
        for v in remaining:
            fc = fill_in_count(v, work_adj)
            tie = -sum(self.qubit_interaction_graph[v].get(u, 0) if isinstance(self.qubit_interaction_graph[v], dict) else self.qubit_interaction_graph[v][u] for u in work_adj[v])
            score = (fc, tie, v)
            if best_score is None or score < best_score:
                best_score = score
                best_v = v
        v = best_v
        nbrs = list(work_adj[v])
        bag = set(nbrs) | {v}
        bags.append((v, bag))
        elim_order.append(v)
        # Connect neighbors (chordal completion)
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                a, b = nbrs[i], nbrs[j]
                work_adj[a].add(b)
                work_adj[b].add(a)
        # Remove v
        for u in nbrs:
            work_adj[u].discard(v)
        del work_adj[v]
        remaining.discard(v)

    # ---- 3. Build bag tree (parent = earliest later bag sharing a vertex) ----
    pos_in_order = {v: i for i, (v, _) in enumerate(bags)}
    children = defaultdict(list)
    parent = [-1] * len(bags)
    roots = []
    for i, (v, bag) in enumerate(bags):
        # find earliest later bag (j > i) whose eliminated vertex is in bag
        candidate = -1
        best_idx = None
        for u in bag:
            if u == v:
                continue
            j = pos_in_order.get(u, -1)
            if j > i and (best_idx is None or j < best_idx):
                best_idx = j
                candidate = j
        parent[i] = candidate
        if candidate == -1:
            roots.append(i)
        else:
            children[candidate].append(i)

    # ---- 4. Hardware embedding: BFS over bag tree, greedy dense placement ----
    placed_logical = {}     # logical -> physical
    used_phys = set()

    def central_unused():
        best_p = None
        best_c = -1.0
        for p in range(N):
            if p in used_phys:
                continue
            c = self.physical_centrality.get(p, 0.0) if isinstance(self.physical_centrality, dict) else 0.0
            if c > best_c:
                best_c = c
                best_p = p
        if best_p is None:
            for p in range(N):
                if p not in used_phys:
                    return p
        return best_p

    def place_logical(lq, phys):
        placed_logical[lq] = phys
        used_phys.add(phys)

    def grow_dense_region(seed_phys, size, anchor_phys_set=None):
        # Greedy expansion preferring high-degree, low-distance, central, adjacent to anchors
        region = []
        if seed_phys is None or seed_phys in used_phys:
            seed_phys = central_unused()
        if seed_phys is None:
            return region
        region.append(seed_phys)
        chosen = {seed_phys}
        while len(region) < size:
            best_p = None
            best_score = None
            for r in region:
                for nb in self.backend[r]:
                    if nb in used_phys or nb in chosen:
                        continue
                    # score: connections to current region + centrality - distance to anchors
                    conn = sum(1 for x in region if nb in self.backend[x])
                    cent = self.physical_centrality.get(nb, 0.0) if isinstance(self.physical_centrality, dict) else 0.0
                    anchor_dist = 0.0
                    if anchor_phys_set:
                        anchor_dist = sum(self.distance_matrix[nb][a] for a in anchor_phys_set)
                    score = (-conn, anchor_dist, -cent, nb)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_p = nb
            if best_p is None:
                # fallback: nearest unused physical to region by distance
                best_p = None
                best_d = None
                for p in range(N):
                    if p in used_phys or p in chosen:
                        continue
                    d = min(self.distance_matrix[p][r] for r in region)
                    if best_d is None or d < best_d:
                        best_d = d
                        best_p = p
                if best_p is None:
                    break
            region.append(best_p)
            chosen.add(best_p)
        return region

    # Process bags via BFS from roots; siblings end up in adjacent regions
    visited_bags = set()
    queue = deque()
    for r in roots:
        queue.append((r, None))  # (bag_idx, parent_separator_phys_set)

    while queue:
        bag_idx, anchor_phys = queue.popleft()
        if bag_idx in visited_bags:
            continue
        visited_bags.add(bag_idx)
        v_elim, bag = bags[bag_idx]

        # separator = bag vertices already placed
        already = [u for u in bag if u in placed_logical]
        to_place = [u for u in bag if u not in placed_logical]

        if not to_place:
            # still enqueue children using this bag's placed phys as anchors
            anchor_set = set(placed_logical[u] for u in bag if u in placed_logical)
            for ch in children[bag_idx]:
                queue.append((ch, anchor_set))
            continue

        # seed: if we have anchors (separator placed), pick adjacent unused; else most central
        seed = None
        anchor_set = set(placed_logical[u] for u in already)
        if anchor_set:
            best_seed = None
            best_score = None
            for a in anchor_set:
                for nb in self.backend[a]:
                    if nb in used_phys:
                        continue
                    cent = self.physical_centrality.get(nb, 0.0) if isinstance(self.physical_centrality, dict) else 0.0
                    score = (-cent, nb)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_seed = nb
            seed = best_seed
        else:
            seed = central_unused()

        region = grow_dense_region(seed, len(to_place), anchor_phys_set=anchor_set if anchor_set else anchor_phys)

        # Order to_place by logical activity (busiest first → most central in region)
        def act(lq):
            return self.logical_activity.get(lq, 0) if isinstance(self.logical_activity, dict) else 0
        to_place.sort(key=lambda lq: -act(lq))

        # Order region by centrality (highest first)
        def reg_key(p):
            return -(self.physical_centrality.get(p, 0.0) if isinstance(self.physical_centrality, dict) else 0.0)
        region.sort(key=reg_key)

        for lq, phys in zip(to_place, region):
            place_logical(lq, phys)

        # If region was too small, fall back for leftovers
        leftovers = [lq for lq in to_place if lq not in placed_logical]
        for lq in leftovers:
            p = central_unused()
            if p is None:
                break
            place_logical(lq, p)

        new_anchor = set(placed_logical[u] for u in bag if u in placed_logical)
        for ch in children[bag_idx]:
            queue.append((ch, new_anchor))

    # ---- 5. Back-fill any remaining logicals from self.access ----
    all_logicals_in_access = set()
    for gid, qs in self.access.items():
        for q in qs:
            all_logicals_in_access.add(q)

    for lq in sorted(all_logicals_in_access):
        if lq in placed_logical:
            continue
        p = central_unused()
        if p is None:
            break
        place_logical(lq, p)

    # ---- 6. Build list-form mapping_dict / reverse_mapping_dict ----
    mapping_dict = [-1] * N
    reverse_mapping_dict = [-1] * N

    for lq, phys in placed_logical.items():
        if 0 <= lq < N and 0 <= phys < N:
            mapping_dict[lq] = phys
            reverse_mapping_dict[phys] = lq

    # Identity-style fill for any remaining logical slots
    unused_phys_list = [p for p in range(N) if p not in used_phys]
    up_idx = 0
    for lq in range(N):
        if mapping_dict[lq] == -1:
            # prefer identity if free
            if lq < N and lq not in used_phys:
                mapping_dict[lq] = lq
                reverse_mapping_dict[lq] = lq
                used_phys.add(lq)
            else:
                while up_idx < len(unused_phys_list) and unused_phys_list[up_idx] in used_phys:
                    up_idx += 1
                if up_idx < len(unused_phys_list):
                    p = unused_phys_list[up_idx]
                    up_idx += 1
                    mapping_dict[lq] = p
                    reverse_mapping_dict[p] = lq
                    used_phys.add(p)

    # Final safety: if any -1 remains in mapping_dict (shouldn't), patch with identity
    if any(x == -1 for x in mapping_dict):
        taken = set(p for p in mapping_dict if p != -1)
        free = [p for p in range(N) if p not in taken]
        fi = 0
        for lq in range(N):
            if mapping_dict[lq] == -1 and fi < len(free):
                mapping_dict[lq] = free[fi]
                reverse_mapping_dict[free[fi]] = lq
                fi += 1

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)