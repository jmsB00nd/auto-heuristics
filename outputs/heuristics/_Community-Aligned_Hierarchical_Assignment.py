def init_mapping(self):
    import numpy as np
    from collections import defaultdict, deque
    from scipy.optimize import linear_sum_assignment

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    n_phys = len(physical_qubits)
    phys_idx = {pq: i for i, pq in enumerate(physical_qubits)}

    # ---------------------------------------------------------------
    # Phase 1: Build temporal-decay weighted interaction graph & detect communities
    # ---------------------------------------------------------------
    # Build DAG for topological ordering (temporal decay)
    schedule = sorted(self.access.keys())
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)

    for node in schedule:
        write_qubits = self.write_dict.get(node, [])
        read_qubits = [q for q in self.access[node] if q not in write_qubits]
        for q in read_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            active_readers[q].add(node)
        for q in write_qubits:
            if q in latest_writer:
                w = latest_writer[q]
                if w != node:
                    successors[w].add(node)
                    predecessors[node].add(w)
            for r in active_readers.get(q, set()):
                if r != node:
                    successors[r].add(node)
                    predecessors[node].add(r)
            active_readers[q].clear()
            latest_writer[q] = node

    # Kahn's topological sort for temporal rank
    all_gates = set(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_rank = {}
    rank = 0
    while queue:
        g = queue.popleft()
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)
    total_gates = max(rank, 1)

    # Build temporal-decay weighted interaction graph
    alpha = 2.0
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            r = topo_rank.get(gate, 0)
            w = np.exp(-alpha * r / total_gates)
            interaction_weight[key] += w
            logical_degree[q1] += w
            logical_degree[q2] += w
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)
    n_logical = len(logical_qubits)
    lq_idx = {lq: i for i, lq in enumerate(logical_qubits)}

    # Build adjacency for logical qubits
    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    # Label propagation community detection on logical interaction graph
    if n_logical <= 1:
        # Trivial case
        communities_logical = {0: logical_qubits[:]}
    else:
        # Determine target number of communities based on graph size
        k_target = max(2, min(n_logical // 4, 8))

        # Label propagation
        labels = {q: i for i, q in enumerate(logical_qubits)}
        max_iter_lp = 30
        for iteration in range(max_iter_lp):
            changed = False
            order = list(logical_qubits)
            np.random.seed(42 + iteration)
            np.random.shuffle(order)
            for q in order:
                if q not in logical_neighbors or not logical_neighbors[q]:
                    continue
                # Weighted vote from neighbors
                label_weights = defaultdict(float)
                for nbr, w in logical_neighbors[q].items():
                    label_weights[labels[nbr]] += w
                if label_weights:
                    best_label = max(label_weights, key=label_weights.get)
                    if best_label != labels[q]:
                        labels[q] = best_label
                        changed = True
            if not changed:
                break

        # Collect communities
        comm_members = defaultdict(list)
        for q in logical_qubits:
            comm_members[labels[q]].append(q)

        # Merge tiny communities (size < 2) into nearest larger community
        communities_list = list(comm_members.values())
        while len(communities_list) > k_target:
            # Merge the two smallest communities
            communities_list.sort(key=len)
            merged = communities_list[0] + communities_list[1]
            communities_list = communities_list[2:] + [merged]

        # If too few communities, split the largest
        while len(communities_list) < k_target and any(len(c) > 2 for c in communities_list):
            communities_list.sort(key=len, reverse=True)
            biggest = communities_list[0]
            if len(biggest) <= 2:
                break
            mid = len(biggest) // 2
            communities_list = [biggest[:mid], biggest[mid:]] + communities_list[1:]

        communities_logical = {i: c for i, c in enumerate(communities_list)}

    k = len(communities_logical)
    # Map each logical qubit to its community
    lq_to_comm = {}
    for ci, members in communities_logical.items():
        for q in members:
            lq_to_comm[q] = ci

    # ---------------------------------------------------------------
    # Phase 2: Partition hardware graph into k regions
    # ---------------------------------------------------------------
    comm_sizes = [len(communities_logical[i]) for i in range(k)]

    if k <= 1:
        # Single community: all physical qubits in one region
        hw_regions = {0: list(physical_qubits)}
    else:
        # Spectral embedding of hardware graph
        D = np.zeros((n_phys, n_phys))
        for i, p1 in enumerate(physical_qubits):
            for j, p2 in enumerate(physical_qubits):
                D[i, j] = self.distance_matrix[p1][p2]

        # Build Laplacian from adjacency
        A_hw = np.zeros((n_phys, n_phys))
        for pq in physical_qubits:
            i = phys_idx[pq]
            for nbr in self.backend.get(pq, []):
                if nbr in phys_idx:
                    j = phys_idx[nbr]
                    A_hw[i, j] = 1.0

        deg = np.sum(A_hw, axis=1)
        L = np.diag(deg) - A_hw

        # Compute first k eigenvectors of Laplacian (skip first trivial one)
        eigvals, eigvecs = np.linalg.eigh(L)
        # Use eigenvectors 1..k for spectral embedding
        n_vecs = min(k, n_phys - 1)
        spectral_coords = eigvecs[:, 1:1 + n_vecs]

        # K-means clustering with size constraints
        # First do standard k-means to get initial centers
        from scipy.cluster.vq import kmeans2
        np.random.seed(42)
        if spectral_coords.shape[1] == 0:
            # Fallback: assign round-robin
            hw_regions = defaultdict(list)
            for i, pq in enumerate(physical_qubits):
                hw_regions[i % k].append(pq)
        else:
            # Run k-means
            try:
                centers, km_labels = kmeans2(spectral_coords, k, minit='points', seed=42)
            except Exception:
                km_labels = np.array([i % k for i in range(n_phys)])

            # Size-constrained reassignment
            # Target sizes: each region should hold at least comm_sizes[i] qubits
            # Sort communities by size descending for assignment
            total_logical = sum(comm_sizes)
            # Each region needs at least comm_sizes[i] physical qubits
            # But we assign communities to regions later, so here just partition roughly equally
            target_sizes = sorted(comm_sizes, reverse=True)
            # Pad to fill all physical qubits
            remaining = n_phys - total_logical
            # Distribute remaining evenly
            base_extra = remaining // k
            leftover = remaining % k
            region_targets = []
            for i in range(k):
                extra = base_extra + (1 if i < leftover else 0)
                region_targets.append(comm_sizes[i] + extra)

            # Sort region_targets descending for greedy assignment
            region_targets_sorted = sorted(enumerate(region_targets), key=lambda x: -x[1])

            # Assign physical qubits to regions greedily using distance to cluster centers
            # Compute distance of each physical qubit to each k-means center
            dists_to_centers = np.zeros((n_phys, k))
            for c in range(k):
                dists_to_centers[:, c] = np.linalg.norm(spectral_coords - centers[c], axis=1)

            assigned = [False] * n_phys
            hw_regions = defaultdict(list)

            for region_id, target_size in region_targets_sorted:
                # Get unassigned qubits sorted by distance to this region's center
                candidates = [(dists_to_centers[i, region_id], i) for i in range(n_phys) if not assigned[i]]
                candidates.sort()
                count = 0
                for _, idx in candidates:
                    if count >= target_size:
                        break
                    hw_regions[region_id].append(physical_qubits[idx])
                    assigned[idx] = True
                    count += 1

            # Assign any remaining unassigned physical qubits to smallest region
            for i in range(n_phys):
                if not assigned[i]:
                    smallest_region = min(hw_regions.keys(), key=lambda r: len(hw_regions[r]))
                    hw_regions[smallest_region].append(physical_qubits[i])
                    assigned[i] = True

    # ---------------------------------------------------------------
    # Phase 3: k×k Hungarian assignment of communities to regions
    # ---------------------------------------------------------------
    # Compute average distance between regions
    region_avg_dist = np.zeros((k, k))
    if k > 1:
        for ri in range(k):
            for rj in range(k):
                if ri == rj:
                    continue
                total_d = 0.0
                count = 0
                for pi in hw_regions[ri]:
                    for pj in hw_regions[rj]:
                        total_d += self.distance_matrix[pi][pj]
                        count += 1
                region_avg_dist[ri][rj] = total_d / max(count, 1)

    # Cost matrix: cost of assigning logical community ci to hardware region rj
    cost_matrix = np.zeros((k, k))
    for ci in range(k):
        members_ci = set(communities_logical[ci])
        for rj in range(k):
            cost = 0.0
            for q1 in communities_logical[ci]:
                for q2_nbr, w in logical_neighbors.get(q1, {}).items():
                    if q2_nbr not in members_ci:
                        # Inter-community interaction
                        comm_q2 = lq_to_comm.get(q2_nbr, ci)
                        # We need avg_dist from rj to all other regions weighted
                        # But since we don't know assignment yet, use avg_dist(rj, all other regions)
                        # weighted by which community q2 belongs to — approximate with uniform
                        # Sum over all possible region assignments for comm_q2
                        # Use a simpler approximation: avg distance from rj to other regions
                        avg_d = 0.0
                        for rk in range(k):
                            if rk != rj:
                                avg_d += region_avg_dist[rj][rk]
                        avg_d /= max(k - 1, 1)
                        cost += w * avg_d
            cost_matrix[ci, rj] = cost

    # Also add intra-community cost: prefer regions with low internal distances for high-interaction communities
    for ci in range(k):
        intra_weight = 0.0
        for q1 in communities_logical[ci]:
            for q2 in communities_logical[ci]:
                if q1 < q2:
                    key = (q1, q2)
                    intra_weight += interaction_weight.get(key, 0.0)
        for rj in range(k):
            # Average internal distance of region rj
            region_pqs = hw_regions[rj]
            if len(region_pqs) > 1:
                avg_internal = 0.0
                cnt = 0
                for pi in region_pqs:
                    for pj in region_pqs:
                        if pi < pj:
                            avg_internal += self.distance_matrix[pi][pj]
                            cnt += 1
                avg_internal /= max(cnt, 1)
            else:
                avg_internal = 0.0
            cost_matrix[ci, rj] += intra_weight * avg_internal

    if k > 1:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        comm_to_region = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    else:
        comm_to_region = {0: 0}

    # ---------------------------------------------------------------
    # Phase 4: Greedy interaction-priority placement within each (community, region) pair
    # ---------------------------------------------------------------
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q
    used_physical = set()

    for ci in range(k):
        ri = comm_to_region[ci]
        members = communities_logical[ci]
        region_pqs = hw_regions[ri]

        if not members:
            continue

        # Available physical qubits in this region
        available = [pq for pq in region_pqs if pq not in used_physical]

        # Sort logical qubits by interaction degree (descending)
        members_sorted = sorted(members, key=lambda q: logical_degree.get(q, 0), reverse=True)

        # Greedy placement: start with highest-degree qubit at most central physical qubit
        if not available:
            continue

        # Find most central physical qubit in region (min sum of distances to others in region)
        if len(available) > 1:
            best_center = min(available, key=lambda pq: sum(self.distance_matrix[pq][opq] for opq in available))
        else:
            best_center = available[0]

        first_lq = members_sorted[0]
        mapping_dict[first_lq] = best_center
        reverse_mapping_dict[best_center] = first_lq
        used_physical.add(best_center)

        placed_in_comm = {first_lq}

        for lq in members_sorted[1:]:
            # Find best physical qubit: minimize weighted distance to already-placed neighbors
            best_pq = None
            best_cost = float('inf')
            for pq in region_pqs:
                if pq in used_physical:
                    continue
                cost = 0.0
                for placed_lq in placed_in_comm:
                    w = logical_neighbors.get(lq, {}).get(placed_lq, 0.0)
                    if w > 0:
                        cost += w * self.distance_matrix[pq][mapping_dict[placed_lq]]
                # Also consider cross-community neighbors already placed
                for nbr, w in logical_neighbors.get(lq, {}).items():
                    if nbr not in placed_in_comm and mapping_dict[nbr] != -1:
                        cost += w * self.distance_matrix[pq][mapping_dict[nbr]]
                if cost < best_cost:
                    best_cost = cost
                    best_pq = pq

            if best_pq is None:
                # Overflow: pick any free physical qubit
                for pq in physical_qubits:
                    if pq not in used_physical:
                        best_pq = pq
                        break

            if best_pq is not None:
                mapping_dict[lq] = best_pq
                reverse_mapping_dict[best_pq] = lq
                used_physical.add(best_pq)
                placed_in_comm.add(lq)

    # ---------------------------------------------------------------
    # Phase 5: Cross-community 2-opt refinement
    # ---------------------------------------------------------------
    def compute_total_cost():
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            if mapping_dict[q1] != -1 and mapping_dict[q2] != -1:
                cost += w * self.distance_matrix[mapping_dict[q1]][mapping_dict[q2]]
        return cost

    # Identify boundary qubits (have inter-community interactions)
    boundary_qubits = set()
    for q in logical_qubits:
        comm_q = lq_to_comm.get(q)
        for nbr in logical_neighbors.get(q, {}):
            if lq_to_comm.get(nbr) != comm_q:
                boundary_qubits.add(q)
                boundary_qubits.add(nbr)
                break

    boundary_list = sorted(boundary_qubits)
    improved = True
    max_rounds = 3
    round_count = 0
    while improved and round_count < max_rounds:
        improved = False
        round_count += 1
        for i in range(len(boundary_list)):
            for j in range(i + 1, len(boundary_list)):
                q1, q2 = boundary_list[i], boundary_list[j]
                if mapping_dict[q1] == -1 or mapping_dict[q2] == -1:
                    continue
                p1, p2 = mapping_dict[q1], mapping_dict[q2]

                # Compute delta cost of swapping
                delta = 0.0
                for nbr, w in logical_neighbors.get(q1, {}).items():
                    if mapping_dict[nbr] == -1:
                        continue
                    pn = mapping_dict[nbr]
                    if nbr == q2:
                        continue
                    delta += w * (self.distance_matrix[p2][pn] - self.distance_matrix[p1][pn])
                for nbr, w in logical_neighbors.get(q2, {}).items():
                    if mapping_dict[nbr] == -1:
                        continue
                    pn = mapping_dict[nbr]
                    if nbr == q1:
                        continue
                    delta += w * (self.distance_matrix[p1][pn] - self.distance_matrix[p2][pn])

                if delta < -1e-9:
                    # Swap
                    mapping_dict[q1], mapping_dict[q2] = p2, p1
                    reverse_mapping_dict[p1] = q2
                    reverse_mapping_dict[p2] = q1
                    improved = True

    # ---------------------------------------------------------------
    # Fill remaining unmapped qubits
    # ---------------------------------------------------------------
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    free_physical = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]
    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)