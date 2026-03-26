def init_mapping(self):
    import random
    from collections import defaultdict, deque

    num_q = self.num_qubits
    dm_size = len(self.distance_matrix)

    # Helper: safe distance lookup
    def dist(p1, p2):
        if p1 < dm_size and p2 < dm_size:
            return self.distance_matrix[p1][p2]
        return float('inf')

    # ── Step 0: Extract 2-qubit gates ──────────────────────────────────
    two_q_gates = []
    for gate_id in sorted(self.access.keys()):
        qubits = self.access[gate_id]
        if len(qubits) == 2:
            two_q_gates.append((gate_id, qubits[0], qubits[1]))

    if not two_q_gates:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ── Step 1: Build approximate DAG layers ───────────────────────────
    layers = []
    used_in_layer = set()
    current_layer = []
    for gid, q1, q2 in two_q_gates:
        if q1 in used_in_layer or q2 in used_in_layer:
            if current_layer:
                layers.append(current_layer)
            current_layer = [(gid, q1, q2)]
            used_in_layer = {q1, q2}
        else:
            current_layer.append((gid, q1, q2))
            used_in_layer.add(q1)
            used_in_layer.add(q2)
    if current_layer:
        layers.append(current_layer)

    L = min(len(layers), 5)

    # Collect front-layer gates with layer weights
    front_gates = []
    for li in range(L):
        for gid, q1, q2 in layers[li]:
            front_gates.append((q1, q2, 1.0 / (li + 1)))

    # Build interaction graph for involved logical qubits
    interaction = defaultdict(lambda: defaultdict(float))
    involved_logical = set()
    for q1, q2, w in front_gates:
        interaction[q1][q2] += w
        interaction[q2][q1] += w
        involved_logical.add(q1)
        involved_logical.add(q2)

    # ── Step 2: Build "ideal" target mapping ───────────────────────────
    # Greedily place interacting logical qubits onto adjacent physical qubits
    lq_sorted = sorted(involved_logical,
                        key=lambda q: sum(interaction[q].values()),
                        reverse=True)

    # Find highest-degree physical qubit as seed
    phys_nodes = sorted(self.backend.keys())
    seed_pq = max(phys_nodes, key=lambda p: len(self.backend[p]))

    ideal = [None] * num_q
    rev_ideal = [None] * num_q
    assigned_phys = set()

    # Place highest-interaction logical qubit at highest-degree physical qubit
    ideal[lq_sorted[0]] = seed_pq
    rev_ideal[seed_pq] = lq_sorted[0]
    assigned_phys.add(seed_pq)
    placed = {lq_sorted[0]}
    queue = deque([lq_sorted[0]])

    while queue and len(placed) < len(lq_sorted):
        cur_lq = queue.popleft()
        cur_pq = ideal[cur_lq]

        # Neighbors in interaction graph, sorted by weight (descending)
        nbrs = sorted(
            [(nq, w) for nq, w in interaction[cur_lq].items() if nq not in placed],
            key=lambda x: -x[1]
        )

        # Available adjacent physical qubits
        avail_adj = [p for p in self.backend[cur_pq] if p not in assigned_phys]

        for nq, _ in nbrs:
            if nq in placed:
                continue
            if avail_adj:
                pq = avail_adj.pop(0)
            else:
                # Closest unassigned physical qubit
                pq = min(
                    (p for p in range(num_q) if p not in assigned_phys),
                    key=lambda p: dist(cur_pq, p)
                )
            ideal[nq] = pq
            rev_ideal[pq] = nq
            assigned_phys.add(pq)
            placed.add(nq)
            queue.append(nq)

    # Place remaining involved qubits not yet placed
    for lq in lq_sorted:
        if lq not in placed:
            for p in range(num_q):
                if p not in assigned_phys:
                    ideal[lq] = p
                    rev_ideal[p] = lq
                    assigned_phys.add(p)
                    placed.add(lq)
                    break

    # Fill uninvolved logical qubits into remaining physical slots
    unassigned_lq = [q for q in range(num_q) if ideal[q] is None]
    unassigned_pq = [p for p in range(num_q) if rev_ideal[p] is None]
    for lq, pq in zip(unassigned_lq, unassigned_pq):
        ideal[lq] = pq
        rev_ideal[pq] = lq

    # ── Step 3: Token swapping lower bound scorer ──────────────────────
    def token_swap_lower_bound(candidate):
        """
        Compute a lower bound on token swapping distance from candidate
        to ideal using cycle decomposition on the hardware graph.

        For each cycle of length k in the displacement permutation,
        LB = max(k-1, ceil(sum_of_edge_distances / 2)).
        This is tighter than simple sum-of-distances.
        """
        # Displacement permutation: token at candidate[q] must go to ideal[q]
        displacement = [0] * num_q
        for q in range(num_q):
            displacement[candidate[q]] = ideal[q]

        visited = [False] * num_q
        total_lb = 0

        for start in range(num_q):
            if visited[start] or displacement[start] == start:
                visited[start] = True
                continue

            # Trace the cycle
            cycle = []
            p = start
            while not visited[p]:
                visited[p] = True
                cycle.append(p)
                p = displacement[p]

            k = len(cycle)
            if k <= 1:
                continue

            # Sum of distances along cycle edges
            cycle_dist = 0
            for i in range(k):
                cycle_dist += dist(cycle[i], cycle[(i + 1) % k])

            # Tight lower bound per cycle
            lb = max(k - 1, (cycle_dist + 1) // 2)
            total_lb += lb

        return total_lb

    # ── Step 4: Generate k candidate mappings ──────────────────────────
    k = 60
    candidates = []

    # Include the ideal mapping itself
    candidates.append(ideal[:])

    # Trivial mapping
    candidates.append(list(range(num_q)))

    # Random permutations
    for _ in range(k - 2):
        perm = list(range(num_q))
        random.shuffle(perm)
        candidates.append(perm)

    # ── Step 5: Select best candidate ──────────────────────────────────
    best_score = float('inf')
    best_mapping = None

    for mapping in candidates:
        score = token_swap_lower_bound(mapping)
        if score < best_score:
            best_score = score
            best_mapping = mapping

    # ── Step 6: Populate mapping dicts ─────────────────────────────────
    self.mapping_dict = best_mapping[:]
    self.reverse_mapping_dict = [0] * num_q
    for logical, physical in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[physical] = logical

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)