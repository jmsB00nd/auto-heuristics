def init_mapping(self):
    from collections import defaultdict, deque
    import math

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())

    # ---------------------------------------------------------------
    # Step 1: Build DAG, topological sort, critical path
    # ---------------------------------------------------------------
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    latest_writer = {}
    active_readers = defaultdict(set)

    for node in sorted(self.access.keys()):
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

    all_gates = set(self.access.keys())
    in_degree = {g: len(predecessors.get(g, set())) for g in all_gates}
    queue = deque(sorted(g for g in all_gates if in_degree[g] == 0))
    topo_order = []
    topo_rank = {}
    gate_layer = {g: 0 for g in all_gates}
    rank = 0

    while queue:
        g = queue.popleft()
        topo_order.append(g)
        topo_rank[g] = rank
        rank += 1
        for s in sorted(successors.get(g, set())):
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            in_degree[s] -= 1
            if in_degree[s] == 0:
                queue.append(s)

    total_gates = max(rank, 1)

    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # ---------------------------------------------------------------
    # Step 2: Build interaction weights (critical-path weighted)
    # ---------------------------------------------------------------
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)
    logical_qubits_set = set()
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        logical_qubits_set.add(q1)
        logical_qubits_set.add(q2)
        key = (min(q1, q2), max(q1, q2))
        cp = critical_path[g] + 1
        layer = gate_layer[g]
        w = cp * (max_layer - layer + 1)
        interaction_weight[key] += w
        logical_degree[q1] += w
        logical_degree[q2] += w

    for g in all_gates:
        if len(self.access[g]) == 1:
            logical_qubits_set.add(self.access[g][0])

    logical_qubits = sorted(logical_qubits_set)
    logical_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        logical_neighbors[q1][q2] = w
        logical_neighbors[q2][q1] = w

    if not logical_qubits:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ===============================================================
    # Phase 1: Quotient graph construction
    # Identify backbone (degree>=3), chain (degree=2), leaf (degree=1)
    # ===============================================================
    phys_deg = {pq: len(self.backend[pq]) for pq in physical_qubits}

    backbone_nodes = sorted([pq for pq in physical_qubits if phys_deg[pq] >= 3])
    chain_nodes_set = set(pq for pq in physical_qubits if phys_deg[pq] == 2)
    leaf_nodes_set = set(pq for pq in physical_qubits if phys_deg[pq] <= 1)

    # Fallback for non-heavy-hex topologies
    if len(backbone_nodes) < 2:
        backbone_nodes = sorted(physical_qubits)
        chain_nodes_set = set()
        leaf_nodes_set = set()

    backbone_set = set(backbone_nodes)

    # Trace chains: paths of degree-2 nodes between backbone nodes
    chains = []  # (bb_start, bb_end_or_None, [chain_path_nodes])
    visited_chain = set()

    for bb in backbone_nodes:
        for neighbor in sorted(self.backend[bb]):
            if neighbor not in backbone_set and neighbor not in visited_chain:
                chain_path = []
                current = neighbor
                prev = bb
                while current not in backbone_set:
                    chain_path.append(current)
                    visited_chain.add(current)
                    next_nodes = [n for n in self.backend[current] if n != prev]
                    if not next_nodes:
                        break
                    prev = current
                    current = next_nodes[0]
                    if current in visited_chain and current not in backbone_set:
                        break

                if current in backbone_set:
                    chains.append((bb, current, chain_path))
                else:
                    chains.append((bb, None, chain_path))

    # Assign chain nodes to backbone regions (split between endpoints)
    bb_region = defaultdict(list)  # backbone_node -> [nearby physical nodes]
    for bb in backbone_nodes:
        bb_region[bb] = []  # backbone node itself is always slot 0

    for bb_start, bb_end, chain_path in chains:
        if bb_end is not None and len(chain_path) > 0:
            mid = (len(chain_path) + 1) // 2
            for cn in chain_path[:mid]:
                bb_region[bb_start].append(cn)
            for cn in chain_path[mid:]:
                bb_region[bb_end].append(cn)
        elif chain_path:
            for cn in chain_path:
                bb_region[bb_start].append(cn)

    # Capacity per backbone region: 1 (backbone itself) + chain slots
    bb_capacity = {bb: 1 + len(bb_region[bb]) for bb in backbone_nodes}

    # Build quotient distance matrix using hardware distances between backbone nodes
    n_bb = len(backbone_nodes)
    bb_index = {bb: i for i, bb in enumerate(backbone_nodes)}
    q_dist = [[0.0] * n_bb for _ in range(n_bb)]
    for i in range(n_bb):
        for j in range(n_bb):
            q_dist[i][j] = self.distance_matrix[backbone_nodes[i]][backbone_nodes[j]]

    # ===============================================================
    # Phase 2: Cluster logical qubits into n_bb groups
    # Capacity-aware greedy clustering
    # ===============================================================
    n_logical = len(logical_qubits)
    n_clusters = min(n_bb, n_logical)

    sorted_logical = sorted(logical_qubits, key=lambda q: logical_degree.get(q, 0), reverse=True)
    cluster_seeds = sorted_logical[:n_clusters]

    cluster_assignment = {}
    clusters = [[] for _ in range(n_clusters)]
    cluster_max_cap = [0] * n_clusters  # will be set after QAP

    for i, seed in enumerate(cluster_seeds):
        cluster_assignment[seed] = i
        clusters[i].append(seed)

    remaining_logical = [q for q in sorted_logical if q not in cluster_assignment]
    for lq in remaining_logical:
        best_cluster = 0
        best_score = -1.0
        for ci in range(n_clusters):
            interaction = sum(logical_neighbors[lq].get(cq, 0.0) for cq in clusters[ci])
            if interaction > best_score:
                best_score = interaction
                best_cluster = ci
        cluster_assignment[lq] = best_cluster
        clusters[best_cluster].append(lq)

    # Build cluster interaction matrix
    cluster_interaction = [[0.0] * n_clusters for _ in range(n_clusters)]
    for (q1, q2), w in interaction_weight.items():
        c1 = cluster_assignment.get(q1, -1)
        c2 = cluster_assignment.get(q2, -1)
        if c1 >= 0 and c2 >= 0 and c1 != c2:
            cluster_interaction[c1][c2] += w
            cluster_interaction[c2][c1] += w

    # ===============================================================
    # Phase 2b: Solve QAP — assign clusters to backbone nodes
    # Greedy + 2-opt local search
    # ===============================================================
    cluster_total = [sum(cluster_interaction[i]) for i in range(n_clusters)]
    sorted_clusters = sorted(range(n_clusters), key=lambda c: cluster_total[c], reverse=True)

    bb_centrality = [sum(q_dist[i]) for i in range(n_bb)]

    cluster_to_bb = [-1] * n_clusters
    bb_used = set()

    for ci in sorted_clusters:
        best_bi = None
        best_cost = float('inf')
        for bi in range(n_bb):
            if bi in bb_used:
                continue
            cost = 0.0
            has_placed = False
            for cj in range(n_clusters):
                if cluster_to_bb[cj] >= 0:
                    cost += cluster_interaction[ci][cj] * q_dist[bi][cluster_to_bb[cj]]
                    has_placed = True
            if not has_placed:
                cost = bb_centrality[bi]
            if cost < best_cost:
                best_cost = cost
                best_bi = bi
        cluster_to_bb[ci] = best_bi
        bb_used.add(best_bi)

    # 2-opt local search on QAP assignment
    improved = True
    rounds = 0
    while improved and rounds < 15:
        improved = False
        rounds += 1
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                bi, bj = cluster_to_bb[i], cluster_to_bb[j]
                if bi < 0 or bj < 0:
                    continue
                delta = 0.0
                for k in range(n_clusters):
                    if k == i or k == j:
                        continue
                    bk = cluster_to_bb[k]
                    if bk < 0:
                        continue
                    delta += cluster_interaction[i][k] * (q_dist[bj][bk] - q_dist[bi][bk])
                    delta += cluster_interaction[j][k] * (q_dist[bi][bk] - q_dist[bj][bk])
                if delta < -1e-12:
                    cluster_to_bb[i] = bj
                    cluster_to_bb[j] = bi
                    improved = True

    # Try swapping with unused backbone nodes
    unused_bb = [bi for bi in range(n_bb) if bi not in set(cluster_to_bb)]
    for ci in range(n_clusters):
        bi_curr = cluster_to_bb[ci]
        best_delta = 0.0
        best_new = None
        for bi_new in unused_bb:
            delta = 0.0
            for cj in range(n_clusters):
                if cj == ci:
                    continue
                bj = cluster_to_bb[cj]
                if bj < 0:
                    continue
                delta += cluster_interaction[ci][cj] * (q_dist[bi_new][bj] - q_dist[bi_curr][bj])
            if delta < best_delta - 1e-12:
                best_delta = delta
                best_new = bi_new
        if best_new is not None:
            cluster_to_bb[ci] = best_new
            unused_bb.remove(best_new)
            unused_bb.append(bi_curr)

    # ===============================================================
    # Phase 3: Place logical qubits onto physical positions
    # Each cluster → backbone region (backbone node + chain slots)
    # Within region: highest-degree qubit → backbone node, rest → chains
    # ===============================================================
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q
    used_physical = set()

    overflow = []  # qubits that don't fit in their assigned region

    for ci in range(n_clusters):
        bi = cluster_to_bb[ci]
        if bi < 0:
            overflow.extend(clusters[ci])
            continue

        bb_phys = backbone_nodes[bi]
        region_slots = [bb_phys] + list(bb_region[bb_phys])

        # Sort cluster members by degree (highest first → backbone position)
        members = sorted(clusters[ci], key=lambda q: logical_degree.get(q, 0), reverse=True)

        # Optimal chain fill: for small chains, try all orderings
        # For simplicity, place by degree (best qubit at hub)
        slot_idx = 0
        for lq in members:
            while slot_idx < len(region_slots) and region_slots[slot_idx] in used_physical:
                slot_idx += 1
            if slot_idx >= len(region_slots):
                overflow.append(lq)
                continue
            pq = region_slots[slot_idx]
            mapping_dict[lq] = pq
            reverse_mapping_dict[pq] = lq
            used_physical.add(pq)
            slot_idx += 1

    # Place overflow qubits at closest free positions to their cluster's backbone
    free_physical = sorted([pq for pq in physical_qubits if pq not in used_physical])
    for lq in overflow:
        ci = cluster_assignment.get(lq, 0)
        bi = cluster_to_bb[ci] if ci < n_clusters and cluster_to_bb[ci] >= 0 else 0
        target = backbone_nodes[bi] if bi < n_bb else physical_qubits[0]

        best_pq = None
        best_dist = float('inf')
        for pq in free_physical:
            d = self.distance_matrix[pq][target]
            if d < best_dist:
                best_dist = d
                best_pq = pq
        if best_pq is not None:
            mapping_dict[lq] = best_pq
            reverse_mapping_dict[best_pq] = lq
            used_physical.add(best_pq)
            free_physical.remove(best_pq)

    # Fill unmapped (non-circuit qubits)
    unmapped = [q for q in range(num_q) if mapping_dict[q] == -1]
    free = [pq for pq in range(num_q) if reverse_mapping_dict[pq] == -1]
    for lq, pq in zip(unmapped, free):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq

    # ===============================================================
    # Phase 4: Pairwise swap refinement
    # ===============================================================
    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    def run_swap_refinement(m, rm, nbrs, max_rounds=5):
        if len(interacting_logical) <= 1:
            return
        improved = True
        rnd = 0
        while improved and rnd < max_rounds:
            improved = False
            rnd += 1
            for i in range(len(interacting_logical)):
                for j in range(i + 1, len(interacting_logical)):
                    lq_a = interacting_logical[i]
                    lq_b = interacting_logical[j]
                    pq_a = m[lq_a]
                    pq_b = m[lq_b]
                    delta = 0.0
                    affected = set(nbrs[lq_a].keys()) | set(nbrs[lq_b].keys())
                    for q in affected:
                        if q == lq_a or q == lq_b:
                            continue
                        pq_q = m[q]
                        w_a = nbrs[lq_a].get(q, 0.0)
                        if w_a > 0:
                            delta += w_a * (self.distance_matrix[pq_b][pq_q] - self.distance_matrix[pq_a][pq_q])
                        w_b = nbrs[lq_b].get(q, 0.0)
                        if w_b > 0:
                            delta += w_b * (self.distance_matrix[pq_a][pq_q] - self.distance_matrix[pq_b][pq_q])
                    if delta < -1e-12:
                        m[lq_a] = pq_b
                        m[lq_b] = pq_a
                        rm[pq_a] = lq_b
                        rm[pq_b] = lq_a
                        improved = True

    run_swap_refinement(mapping_dict, reverse_mapping_dict, logical_neighbors)

    # ===============================================================
    # Phase 4b: RSDIWR refinement with routing simulation
    # ===============================================================
    last_2q_on_qubit = {}
    gates_2q = {}
    dag2q_succ = defaultdict(set)
    dag2q_pred = defaultdict(set)

    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits
            gates_2q[gate] = (q1, q2)
            for q in [q1, q2]:
                if q in last_2q_on_qubit:
                    prev = last_2q_on_qubit[q]
                    if prev != gate:
                        dag2q_succ[prev].add(gate)
                        dag2q_pred[gate].add(prev)
                last_2q_on_qubit[q] = gate

    topo_2q = []
    in_deg_2q = {g: len(dag2q_pred[g]) for g in gates_2q}
    q2_queue = deque(sorted(g for g in gates_2q if in_deg_2q[g] == 0))
    while q2_queue:
        g = q2_queue.popleft()
        topo_2q.append(g)
        for s in dag2q_succ[g]:
            in_deg_2q[s] -= 1
            if in_deg_2q[s] == 0:
                q2_queue.append(s)

    dep_count_2q = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count_2q[g] += dep_count_2q[s] + 1

    def simulate_routing(m, rm, max_layers=25):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)
        if not gates_2q:
            return swap_counts
        pred_rem = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set(g for g in gates_2q if pred_rem[g] == 0)
        layers_done = 0
        sim_decay = [1.0] * num_q
        while front and layers_done < max_layers:
            executable = []
            for g in front:
                gq1, gq2 = gates_2q[g]
                p1, p2 = sim_m[gq1], sim_m[gq2]
                if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
                    executable.append(g)
            if executable:
                for g in executable:
                    front.discard(g)
                    for s in dag2q_succ[g]:
                        pred_rem[s] -= 1
                        if pred_rem[s] == 0:
                            front.add(s)
                layers_done += 1
                sim_decay = [1.0] * num_q
                continue
            active_phys = set()
            for g in front:
                gq1, gq2 = gates_2q[g]
                active_phys.add(sim_m[gq1])
                active_phys.add(sim_m[gq2])
            candidates = set()
            for pq in active_phys:
                for nb in self.backend.get(pq, []):
                    candidates.add((min(pq, nb), max(pq, nb)))
            best_swap = None
            best_score = float('inf')
            for (s1, s2) in candidates:
                l1 = sim_rm[s1]
                l2 = sim_rm[s2]
                max_d = max(sim_decay[s1], sim_decay[s2])
                score = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1 = sim_m[gq1]
                    p2 = sim_m[gq2]
                    if gq1 == l1: p1 = s2
                    elif gq1 == l2: p1 = s1
                    if gq2 == l1: p2 = s2
                    elif gq2 == l2: p2 = s1
                    deps = dep_count_2q.get(g, 0) + 1
                    score += deps * self.distance_matrix[p1][p2]
                score *= max_d
                if score < best_score:
                    best_score = score
                    best_swap = (s1, s2)
            if best_swap is None:
                break
            s1, s2 = best_swap
            l1 = sim_rm[s1]
            l2 = sim_rm[s2]
            sim_m[l1] = s2
            sim_m[l2] = s1
            sim_rm[s1] = l2
            sim_rm[s2] = l1
            sim_decay[s1] += 0.001
            sim_decay[s2] += 0.001
            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0
        return swap_counts

    def compute_total_cost(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost

    # RSDIWR: 3 iterations with decaying alpha
    alpha_schedule = [1.0, 0.55, 0.20]
    swap_counts = defaultdict(float)
    best_m = list(mapping_dict)
    best_rm = list(reverse_mapping_dict)
    best_cost = compute_total_cost(mapping_dict, interaction_weight)

    for t in range(3):
        alpha = alpha_schedule[t]
        if t == 0 or not swap_counts:
            eff_weights = dict(interaction_weight)
        else:
            eff_weights = defaultdict(float)
            max_s = max(interaction_weight.values()) if interaction_weight else 1.0
            max_sw = max(swap_counts.values()) if swap_counts else 1.0
            scale = max_s / max(max_sw, 1e-10)
            all_keys = set(interaction_weight.keys()) | set(swap_counts.keys())
            for key in all_keys:
                w_s = interaction_weight.get(key, 0.0)
                w_r = swap_counts.get(key, 0.0) * scale
                eff_weights[key] = alpha * w_s + (1 - alpha) * w_r

        eff_nbrs = defaultdict(dict)
        for (eq1, eq2), w in eff_weights.items():
            eff_nbrs[eq1][eq2] = w
            eff_nbrs[eq2][eq1] = w

        m_try = list(best_m)
        rm_try = list(best_rm)
        run_swap_refinement(m_try, rm_try, eff_nbrs, max_rounds=4)

        cost = compute_total_cost(m_try, interaction_weight)
        if cost < best_cost:
            best_cost = cost
            best_m = list(m_try)
            best_rm = list(rm_try)

        if t < 2:
            swap_counts = simulate_routing(best_m, best_rm, max_layers=25)

    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)