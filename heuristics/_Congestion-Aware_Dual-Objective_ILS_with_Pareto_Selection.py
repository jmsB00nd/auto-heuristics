def init_mapping(self):
    import math
    import random
    from collections import defaultdict, deque

    num_q = self.num_qubits
    dm = self.distance_matrix
    backend = self.backend
    backend_conn = self.backend_connections

    # --- Step 1: Build DAG, topological order, interaction weights ---
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    last_write = {}
    last_read = defaultdict(set)
    all_gates = list(self.access.keys())

    for gate_id in all_gates:
        qubits = self.access[gate_id]
        write_qubits = self.write_dict.get(gate_id, [])
        read_qubits = [q for q in qubits if q not in write_qubits]
        for q in qubits:
            if q in last_write:
                predecessors[gate_id].add(last_write[q])
                successors[last_write[q]].add(gate_id)
        for q in read_qubits:
            if q in last_write:
                predecessors[gate_id].add(last_write[q])
                successors[last_write[q]].add(gate_id)
        for q in write_qubits:
            for prev_reader in last_read[q]:
                if prev_reader != gate_id:
                    predecessors[gate_id].add(prev_reader)
                    successors[prev_reader].add(gate_id)
            last_write[q] = gate_id
            last_read[q] = set()
        for q in read_qubits:
            last_read[q].add(gate_id)

    # Topological sort via Kahn's
    in_deg = defaultdict(int)
    for g in all_gates:
        in_deg[g] = len(predecessors[g])
    queue = deque([g for g in all_gates if in_deg[g] == 0])
    topo_order = []
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in successors[g]:
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)

    topo_rank = {g: i for i, g in enumerate(topo_order)}
    total_gates = max(len(topo_order), 1)

    # Build interaction weights with temporal decay
    alpha = 2.5
    interaction_weight = defaultdict(float)
    logical_qubits_used = set()
    two_q_gates = []

    for gate_id in all_gates:
        qubits = self.access[gate_id]
        for q in qubits:
            logical_qubits_used.add(q)
        if len(qubits) == 2:
            two_q_gates.append(gate_id)
            q1, q2 = qubits
            rank = topo_rank.get(gate_id, 0)
            w = math.exp(-alpha * rank / total_gates)
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += w

    logical_qubits_used = sorted(logical_qubits_used)
    n_logical = len(logical_qubits_used)
    physical_qubits = sorted(backend.keys())

    # Build logical neighbor structure
    logical_neighbors = defaultdict(lambda: defaultdict(float))
    logical_degree = defaultdict(float)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] += w
        logical_neighbors[q2][q1] += w
        logical_degree[q1] += w
        logical_degree[q2] += w

    # Physical centrality
    centrality = {}
    for pq in physical_qubits:
        centrality[pq] = sum(dm[pq][pq2] for pq2 in physical_qubits)

    hw_adj = {pq: set(backend.get(pq, [])) for pq in physical_qubits}

    # --- Step 2: BFS shortest path edges precomputation ---
    # For each pair (p1, p2), store the edges on a shortest path
    def bfs_path_edges(src, dst):
        """Return set of edges (as frozensets) on a shortest path from src to dst."""
        if src == dst:
            return []
        visited = {src}
        parent = {src: None}
        q = deque([src])
        while q:
            node = q.popleft()
            if node == dst:
                break
            for nb in backend.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    q.append(nb)
        # Reconstruct path edges
        edges = []
        cur = dst
        while parent.get(cur) is not None:
            prev = parent[cur]
            edges.append((min(prev, cur), max(prev, cur)))
            cur = prev
        return edges

    # Precompute path edges between all physical pairs that might matter
    # Cache BFS path edges
    path_edge_cache = {}
    def get_path_edges(p1, p2):
        key = (min(p1, p2), max(p1, p2))
        if key not in path_edge_cache:
            path_edge_cache[key] = bfs_path_edges(p1, p2)
        return path_edge_cache[key]

    # --- Step 3: Objective functions ---
    interaction_pairs = list(interaction_weight.keys())

    def compute_objectives(mapping):
        """Return (total_weighted_distance, max_edge_congestion)."""
        total_wd = 0.0
        edge_load = defaultdict(float)
        for (l1, l2), w in interaction_weight.items():
            p1, p2 = mapping[l1], mapping[l2]
            total_wd += w * dm[p1][p2]
            # Compute edge loads
            for edge in get_path_edges(p1, p2):
                edge_load[edge] += w
        max_cong = max(edge_load.values()) if edge_load else 0.0
        return total_wd, max_cong

    def dominates(obj_a, obj_b):
        """True if a dominates b (both objectives <= and at least one <)."""
        return (obj_a[0] <= obj_b[0] and obj_a[1] <= obj_b[1] and
                (obj_a[0] < obj_b[0] or obj_a[1] < obj_b[1]))

    # --- Step 4: Greedy multi-seed construction ---
    def greedy_construct(logical_seed, physical_seed):
        mapping = [-1] * num_q
        rev_mapping = [-1] * num_q
        placed = set()
        placed_phys = set()

        mapping[logical_seed] = physical_seed
        rev_mapping[physical_seed] = logical_seed
        placed.add(logical_seed)
        placed_phys.add(physical_seed)

        while len(placed) < n_logical:
            best_lq, best_pq, best_score = -1, -1, float('inf')
            # Find unplaced logical qubit with highest interaction to placed
            candidates = []
            for lq in logical_qubits_used:
                if lq in placed:
                    continue
                interact = sum(logical_neighbors[lq].get(plq, 0.0) for plq in placed)
                candidates.append((lq, interact))
            candidates.sort(key=lambda x: -x[1])

            for lq, _ in candidates[:3]:
                for pq in physical_qubits:
                    if pq in placed_phys:
                        continue
                    score = 0.0
                    for plq in placed:
                        w = logical_neighbors[lq].get(plq, 0.0)
                        if w > 0:
                            d = dm[pq][mapping[plq]]
                            # adjacency bonus
                            if pq in hw_adj.get(mapping[plq], set()):
                                d *= 0.87
                            score += w * d
                    if score < best_score:
                        best_score = score
                        best_lq = lq
                        best_pq = pq

            if best_lq == -1:
                break
            mapping[best_lq] = best_pq
            rev_mapping[best_pq] = best_lq
            placed.add(best_lq)
            placed_phys.add(best_pq)

        # Fill unmapped
        unmapped_lq = [q for q in range(num_q) if mapping[q] == -1]
        free_pq = [q for q in range(num_q) if rev_mapping[q] == -1]
        for lq, pq in zip(unmapped_lq, free_pq):
            mapping[lq] = pq
            rev_mapping[pq] = lq

        return mapping, rev_mapping

    # Select seeds
    sorted_logical = sorted(logical_qubits_used, key=lambda q: -logical_degree.get(q, 0))
    top_logical = sorted_logical[:min(3, len(sorted_logical))]
    sorted_physical = sorted(physical_qubits, key=lambda pq: centrality.get(pq, float('inf')))
    top_physical = sorted_physical[:min(3, len(sorted_physical))]

    # --- Step 5: Local search (swap refinement) ---
    def local_search(mapping, rev_mapping):
        improved = True
        rounds = 0
        while improved and rounds < 4:
            improved = False
            rounds += 1
            for (l1, l2) in interaction_pairs:
                p1, p2 = mapping[l1], mapping[l2]
                # Compute delta cost of swapping physical assignments of l1 and l2
                delta = 0.0
                for nb, w in logical_neighbors[l1].items():
                    if nb == l2:
                        continue
                    pnb = mapping[nb]
                    delta += w * (dm[p2][pnb] - dm[p1][pnb])
                for nb, w in logical_neighbors[l2].items():
                    if nb == l1:
                        continue
                    pnb = mapping[nb]
                    delta += w * (dm[p1][pnb] - dm[p2][pnb])
                if delta < -1e-12:
                    mapping[l1], mapping[l2] = p2, p1
                    rev_mapping[p1], rev_mapping[p2] = l2, l1
                    improved = True
        return mapping, rev_mapping

    # --- Step 6: ILS + SA with Pareto front ---
    best_mappings = []  # List of (obj, mapping, rev_mapping)
    pareto_front = []   # Non-dominated solutions

    def add_to_pareto(obj, mapping, rev_mapping):
        nonlocal pareto_front
        # Remove dominated solutions
        new_front = []
        is_dominated = False
        for (o, m, r) in pareto_front:
            if dominates(o, obj):
                is_dominated = True
                new_front.append((o, m, r))
            elif not dominates(obj, o):
                new_front.append((o, m, r))
        if not is_dominated:
            new_front.append((obj, mapping[:], rev_mapping[:]))
        pareto_front = new_front

    # Multi-seed construction + refinement
    for ls in top_logical:
        for ps in top_physical:
            m, r = greedy_construct(ls, ps)
            m, r = local_search(m, r)
            obj = compute_objectives(m)
            add_to_pareto(obj, m, r)
            best_mappings.append((obj, m[:], r[:]))

    # Pick best single-objective solution as SA starting point
    best_mappings.sort(key=lambda x: x[0][0])
    current_mapping = best_mappings[0][1][:]
    current_rev = best_mappings[0][2][:]
    current_obj = best_mappings[0][0]

    # SA + ILS parameters
    T = 1.0
    T_min = 0.005
    cooling = 0.97
    max_iters_per_temp = max(50, n_logical * 2)

    def perturb(mapping, rev_mapping, mode):
        m = mapping[:]
        r = rev_mapping[:]
        if mode == 0:
            # Edge-targeted: swap physical qubits of highest-cost pair
            worst_cost = -1
            worst_pair = None
            for (l1, l2), w in interaction_weight.items():
                c = w * dm[m[l1]][m[l2]]
                if c > worst_cost:
                    worst_cost = c
                    worst_pair = (l1, l2)
            if worst_pair:
                l1, l2 = worst_pair
                p1, p2 = m[l1], m[l2]
                # Try swapping l1 with a neighbor of p2
                neighbors_p2 = list(hw_adj.get(p2, []))
                if neighbors_p2:
                    target_p = random.choice(neighbors_p2)
                    target_l = r[target_p]
                    m[l1], m[target_l] = m[target_l], m[l1]
                    r[m[l1]], r[m[target_l]] = l1, target_l
        elif mode == 1:
            # Random double swap
            if n_logical >= 4:
                sample = random.sample(logical_qubits_used, 4)
                p0, p1, p2, p3 = m[sample[0]], m[sample[1]], m[sample[2]], m[sample[3]]
                m[sample[0]], m[sample[1]] = p1, p0
                r[p0], r[p1] = sample[1], sample[0]
                m[sample[2]], m[sample[3]] = p3, p2
                r[p2], r[p3] = sample[3], sample[2]
            elif n_logical >= 2:
                s = random.sample(logical_qubits_used, 2)
                p0, p1 = m[s[0]], m[s[1]]
                m[s[0]], m[s[1]] = p1, p0
                r[p0], r[p1] = s[1], s[0]
        else:
            # Random single swap
            if n_logical >= 2:
                s = random.sample(logical_qubits_used, 2)
                p0, p1 = m[s[0]], m[s[1]]
                m[s[0]], m[s[1]] = p1, p0
                r[p0], r[p1] = s[1], s[0]
        return m, r

    # ILS + SA loop
    iteration = 0
    while T > T_min:
        for _ in range(max_iters_per_temp):
            mode = random.choice([0, 1, 2])
            cand_m, cand_r = perturb(current_mapping, current_rev, mode)
            cand_m, cand_r = local_search(cand_m, cand_r)
            cand_obj = compute_objectives(cand_m)

            add_to_pareto(cand_obj, cand_m, cand_r)

            # SA acceptance: accept if dominates on either objective or with SA probability
            if dominates(cand_obj, current_obj):
                current_mapping = cand_m
                current_rev = cand_r
                current_obj = cand_obj
            elif cand_obj[0] < current_obj[0] or cand_obj[1] < current_obj[1]:
                current_mapping = cand_m
                current_rev = cand_r
                current_obj = cand_obj
            else:
                delta = (cand_obj[0] - current_obj[0])
                if T > 0 and random.random() < math.exp(-delta / (T * max(current_obj[0], 1e-9))):
                    current_mapping = cand_m
                    current_rev = cand_r
                    current_obj = cand_obj

            iteration += 1
            if iteration > 800:
                break
        if iteration > 800:
            break
        T *= cooling

    # --- Step 7: Select from Pareto front via fast routing simulation ---
    def simulate_routing(mapping, num_layers=5):
        """Fast routing simulation: count swaps needed for first num_layers of 2q gates."""
        sim_map = mapping[:]
        sim_rev = [0] * num_q
        for i in range(num_q):
            sim_rev[sim_map[i]] = i

        swap_count = 0
        gates_done = 0

        # Get gates in topological order
        for gate_id in topo_order:
            qubits = self.access[gate_id]
            if len(qubits) != 2:
                continue
            l1, l2 = qubits
            p1, p2 = sim_map[l1], sim_map[l2]

            # Route toward each other using greedy swaps
            while dm[p1][p2] > 1:
                # Move p1 toward p2
                best_nb = None
                best_d = dm[p1][p2]
                for nb in backend.get(p1, []):
                    if dm[nb][p2] < best_d:
                        best_d = dm[nb][p2]
                        best_nb = nb
                if best_nb is None:
                    break
                # Swap p1 and best_nb
                l_at_nb = sim_rev[best_nb]
                l_at_p1 = sim_rev[p1]
                sim_map[l_at_p1] = best_nb
                sim_map[l_at_nb] = p1
                sim_rev[best_nb] = l_at_p1
                sim_rev[p1] = l_at_nb
                p1 = best_nb
                swap_count += 1

            gates_done += 1
            if gates_done >= num_layers:
                break

        return swap_count

    # Pick best from Pareto front
    if len(pareto_front) > 0:
        best_swaps = float('inf')
        best_m, best_r = None, None
        for (obj, m, r) in pareto_front:
            swaps = simulate_routing(m, num_layers=5)
            if swaps < best_swaps:
                best_swaps = swaps
                best_m = m
                best_r = r
        final_mapping = best_m
        final_rev = best_r
    else:
        final_mapping = current_mapping
        final_rev = current_rev

    # --- Step 8: Assign results ---
    self.mapping_dict = final_mapping
    self.reverse_mapping_dict = final_rev
    self.mapping = final_mapping[:]
    self.reverse_mapping = final_rev[:]

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)