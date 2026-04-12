def init_mapping(self):
    import math
    import random
    import heapq
    from collections import defaultdict, deque
    from time import time as _time

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    dist = self.distance_matrix

    # ---------------------------------------------------------------
    # Step 1: Build DAG and topological layers
    # ---------------------------------------------------------------
    all_gates = sorted(self.access.keys())
    two_qubit_gates = [g for g in all_gates if len(self.access[g]) == 2]

    last_gate_on_qubit = {}
    successors_dag = defaultdict(set)
    predecessors_dag = defaultdict(set)
    for g in all_gates:
        for q in self.access[g]:
            if q in last_gate_on_qubit:
                pred = last_gate_on_qubit[q]
                successors_dag[pred].add(g)
                predecessors_dag[g].add(pred)
            last_gate_on_qubit[q] = g

    in_degree = {g: len(predecessors_dag[g]) for g in all_gates}
    gate_layer = {g: 0 for g in all_gates}
    temp_in = dict(in_degree)
    queue = deque(g for g in all_gates if in_degree[g] == 0)
    topo_order = []
    while queue:
        g = queue.popleft()
        topo_order.append(g)
        for s in successors_dag[g]:
            gate_layer[s] = max(gate_layer[s], gate_layer[g] + 1)
            temp_in[s] -= 1
            if temp_in[s] == 0:
                queue.append(s)

    critical_path = {g: 0 for g in all_gates}
    for g in reversed(topo_order):
        for s in successors_dag[g]:
            if critical_path[s] + 1 > critical_path[g]:
                critical_path[g] = critical_path[s] + 1

    # 2q DAG for routing simulation
    gates_2q = {}
    dag2q_succ = defaultdict(set)
    dag2q_pred = defaultdict(set)
    last_2q_on_qubit = {}
    logical_qubits_set = set()

    for gate in all_gates:
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits
            gates_2q[gate] = (q1, q2)
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            for q in [q1, q2]:
                if q in last_2q_on_qubit:
                    prev = last_2q_on_qubit[q]
                    if prev != gate:
                        dag2q_succ[prev].add(gate)
                        dag2q_pred[gate].add(prev)
                last_2q_on_qubit[q] = gate
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])

    logical_qubits = sorted(logical_qubits_set)

    if not gates_2q:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Topological order for 2q DAG
    topo_2q = []
    in_deg_2q = {g: len(dag2q_pred[g]) for g in gates_2q}
    q2 = deque(sorted(g for g in gates_2q if in_deg_2q[g] == 0))
    while q2:
        g = q2.popleft()
        topo_2q.append(g)
        for s in dag2q_succ[g]:
            in_deg_2q[s] -= 1
            if in_deg_2q[s] == 0:
                q2.append(s)

    dep_count = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count[g] = max(dep_count[g], dep_count[s] + 1)

    bwd_dep = defaultdict(int)
    for g in topo_2q:
        for p in dag2q_pred[g]:
            if p in gates_2q:
                bwd_dep[g] = max(bwd_dep[g], bwd_dep[p] + 1)

    # ---------------------------------------------------------------
    # Step 2: Interaction weights
    # ---------------------------------------------------------------
    max_layer = max((gate_layer[g] for g in two_qubit_gates), default=1)

    static_weight = defaultdict(float)
    logical_degree = defaultdict(float)

    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        cp = critical_path[g] + 1
        layer = gate_layer[g]
        w = cp * (max_layer - layer + 1)
        key = (min(q1, q2), max(q1, q2))
        static_weight[key] += w
        logical_degree[q1] += w
        logical_degree[q2] += w

    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    # ---------------------------------------------------------------
    # Step 3: Physical graph properties
    # ---------------------------------------------------------------
    hw_adj = defaultdict(set)
    for pq in physical_qubits:
        for pq2 in self.backend.get(pq, []):
            hw_adj[pq].add(pq2)

    phys_degree = {pq: len(hw_adj[pq]) for pq in physical_qubits}
    phys_centrality = {}
    for pq in physical_qubits:
        phys_centrality[pq] = sum(dist[pq][pq2] for pq2 in physical_qubits)

    # ---------------------------------------------------------------
    # Step 4: Core helper functions
    # ---------------------------------------------------------------
    def build_neighbors(weights):
        nbrs = defaultdict(dict)
        deg = defaultdict(float)
        for (q1, q2), w in weights.items():
            nbrs[q1][q2] = w
            nbrs[q2][q1] = w
            deg[q1] += w
            deg[q2] += w
        return nbrs, deg

    def compute_cost_nonlinear(m, weights, alpha_exp):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                d = dist[m[q1]][m[q2]]
                cost += w * (d ** alpha_exp)
        return cost

    def compute_cost(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * dist[m[q1]][m[q2]]
        return cost

    def delta_swap_cost_nonlinear(m, rm, pq_a, pq_b, nbrs, alpha_exp):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        delta = 0.0
        affected = set()
        if lq_a in nbrs:
            affected.update(nbrs[lq_a].keys())
        if lq_b in nbrs:
            affected.update(nbrs[lq_b].keys())
        for q in affected:
            if q == lq_a or q == lq_b:
                continue
            pq_q = m[q]
            w_a = nbrs.get(lq_a, {}).get(q, 0.0)
            if w_a > 0:
                old_d = dist[pq_a][pq_q] ** alpha_exp
                new_d = dist[pq_b][pq_q] ** alpha_exp
                delta += w_a * (new_d - old_d)
            w_b = nbrs.get(lq_b, {}).get(q, 0.0)
            if w_b > 0:
                old_d = dist[pq_b][pq_q] ** alpha_exp
                new_d = dist[pq_a][pq_q] ** alpha_exp
                delta += w_b * (new_d - old_d)
        return delta

    def do_swap(m, rm, pq_a, pq_b):
        lq_a, lq_b = rm[pq_a], rm[pq_b]
        m[lq_a], m[lq_b] = pq_b, pq_a
        rm[pq_a], rm[pq_b] = lq_b, lq_a

    def fill_unmapped(m, rm):
        unmapped = [q for q in range(num_q) if m[q] == -1]
        free = [pq for pq in range(num_q) if rm[pq] == -1]
        for lq, pq in zip(unmapped, free):
            m[lq] = pq
            rm[pq] = lq

    # ---------------------------------------------------------------
    # Step 5: Maximum-weight matching on interaction graph
    # ---------------------------------------------------------------
    def greedy_max_weight_matching(weights, nodes):
        """Greedy approximation of maximum-weight matching.
        Returns list of disjoint pairs (q1, q2)."""
        edges = []
        for (q1, q2), w in weights.items():
            if q1 in nodes and q2 in nodes:
                edges.append((w, q1, q2))
        edges.sort(reverse=True)
        matched = set()
        pairs = []
        for w, q1, q2 in edges:
            if q1 not in matched and q2 not in matched:
                pairs.append((q1, q2))
                matched.add(q1)
                matched.add(q2)
        return pairs

    def greedy_max_weight_matching_hw(hw_adj_local, phys_nodes):
        """Greedy matching on hardware graph by combined degree."""
        edges = []
        seen = set()
        for pq in phys_nodes:
            for nb in hw_adj_local.get(pq, set()):
                if nb in phys_nodes:
                    edge = (min(pq, nb), max(pq, nb))
                    if edge not in seen:
                        seen.add(edge)
                        combined_deg = len(hw_adj_local.get(pq, set())) + len(hw_adj_local.get(nb, set()))
                        edges.append((combined_deg, pq, nb))
        edges.sort(reverse=True)
        matched = set()
        pairs = []
        for _, p1, p2 in edges:
            if p1 not in matched and p2 not in matched:
                pairs.append((p1, p2))
                matched.add(p1)
                matched.add(p2)
        return pairs

    # ---------------------------------------------------------------
    # Step 6: Graph contraction functions
    # ---------------------------------------------------------------
    def contract_interaction_graph(weights, logical_pairs, interacting_set):
        """Contract the interaction graph by merging logical pairs into super-qubits.
        Returns: contracted_weights, super_to_pair mapping, pair_to_super mapping."""
        # Map each qubit to its super-qubit ID
        qubit_to_super = {}
        super_to_pair = {}  # super_id -> (q1, q2)
        next_super_id = 0

        for (q1, q2) in logical_pairs:
            qubit_to_super[q1] = next_super_id
            qubit_to_super[q2] = next_super_id
            super_to_pair[next_super_id] = (q1, q2)
            next_super_id += 1

        # Singleton qubits (not in any pair)
        for q in interacting_set:
            if q not in qubit_to_super:
                qubit_to_super[q] = next_super_id
                super_to_pair[next_super_id] = (q,)
                next_super_id += 1

        # Build contracted weights
        contracted_weights = defaultdict(float)
        for (q1, q2), w in weights.items():
            s1 = qubit_to_super.get(q1)
            s2 = qubit_to_super.get(q2)
            if s1 is not None and s2 is not None and s1 != s2:
                key = (min(s1, s2), max(s1, s2))
                contracted_weights[key] += w

        return contracted_weights, super_to_pair, qubit_to_super

    def contract_hardware_graph(hw_pairs, all_phys):
        """Contract hardware graph by merging physical pairs into super-positions.
        Returns: contracted_dist, super_to_phys_pair, phys_to_super, contracted_hw_adj."""
        phys_to_super = {}
        super_to_phys = {}
        next_id = 0

        for (p1, p2) in hw_pairs:
            phys_to_super[p1] = next_id
            phys_to_super[p2] = next_id
            super_to_phys[next_id] = (p1, p2)
            next_id += 1

        for p in all_phys:
            if p not in phys_to_super:
                phys_to_super[p] = next_id
                super_to_phys[next_id] = (p,)
                next_id += 1

        n_super = next_id
        # Contracted distance: min distance between members
        contracted_dist = [[0] * n_super for _ in range(n_super)]
        for i in range(n_super):
            for j in range(n_super):
                if i == j:
                    contracted_dist[i][j] = 0
                else:
                    min_d = float('inf')
                    for pi in super_to_phys[i]:
                        for pj in super_to_phys[j]:
                            d = dist[pi][pj]
                            if d < min_d:
                                min_d = d
                    contracted_dist[i][j] = min_d

        # Contracted adjacency
        contracted_hw_adj = defaultdict(set)
        for i in range(n_super):
            for j in range(n_super):
                if i != j and contracted_dist[i][j] <= 1:
                    contracted_hw_adj[i].add(j)

        return contracted_dist, super_to_phys, phys_to_super, contracted_hw_adj, n_super

    # ---------------------------------------------------------------
    # Step 7: Routing simulation (works on any resolution)
    # ---------------------------------------------------------------
    def simulate_routing_greedy(m, rm, max_layers=20):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)
        total_swaps = 0

        if not gates_2q:
            return swap_counts, total_swaps

        pred_remaining = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set(g for g in gates_2q if pred_remaining[g] == 0)
        layers_done = 0

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
                        pred_remaining[s] -= 1
                        if pred_remaining[s] == 0:
                            front.add(s)
                layers_done += 1
                continue

            active_phys = set()
            for g in front:
                gq1, gq2 = gates_2q[g]
                active_phys.add(sim_m[gq1])
                active_phys.add(sim_m[gq2])

            candidates_sw = set()
            for pq in active_phys:
                for nb in self.backend.get(pq, []):
                    candidates_sw.add((min(pq, nb), max(pq, nb)))

            best_swap, best_score = None, float('inf')
            for (s1, s2) in candidates_sw:
                l1, l2 = sim_rm[s1], sim_rm[s2]
                score = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1, p2 = sim_m[gq1], sim_m[gq2]
                    if gq1 == l1: p1 = s2
                    elif gq1 == l2: p1 = s1
                    if gq2 == l1: p2 = s2
                    elif gq2 == l2: p2 = s1
                    deps = dep_count.get(g, 0) + 1
                    score += deps * dist[p1][p2]
                if score < best_score:
                    best_score = score
                    best_swap = (s1, s2)

            if best_swap is None:
                break

            s1, s2 = best_swap
            l1, l2 = sim_rm[s1], sim_rm[s2]
            sim_m[l1], sim_m[l2] = s2, s1
            sim_rm[s1], sim_rm[s2] = l2, l1
            total_swaps += 1

            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0

        return swap_counts, total_swaps

    def simulate_routing_backward(m, rm, max_layers=20):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)
        total_swaps = 0

        if not gates_2q:
            return swap_counts, total_swaps

        succ_remaining = {g: len(dag2q_succ[g] & set(gates_2q.keys())) for g in gates_2q}
        front = set(g for g in gates_2q if succ_remaining[g] == 0)
        layers_done = 0

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
                    for p in dag2q_pred[g]:
                        if p in gates_2q:
                            succ_remaining[p] -= 1
                            if succ_remaining[p] == 0:
                                front.add(p)
                layers_done += 1
                continue

            active_phys = set()
            for g in front:
                gq1, gq2 = gates_2q[g]
                active_phys.add(sim_m[gq1])
                active_phys.add(sim_m[gq2])

            candidates_sw = set()
            for pq in active_phys:
                for nb in self.backend.get(pq, []):
                    candidates_sw.add((min(pq, nb), max(pq, nb)))

            best_swap, best_score = None, float('inf')
            for (s1, s2) in candidates_sw:
                l1, l2 = sim_rm[s1], sim_rm[s2]
                score = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1, p2 = sim_m[gq1], sim_m[gq2]
                    if gq1 == l1: p1 = s2
                    elif gq1 == l2: p1 = s1
                    if gq2 == l1: p2 = s2
                    elif gq2 == l2: p2 = s1
                    deps = bwd_dep.get(g, 0) + 1
                    score += deps * dist[p1][p2]
                if score < best_score:
                    best_score = score
                    best_swap = (s1, s2)

            if best_swap is None:
                break

            s1, s2 = best_swap
            l1, l2 = sim_rm[s1], sim_rm[s2]
            sim_m[l1], sim_m[l2] = s2, s1
            sim_rm[s1], sim_rm[s2] = l2, l1
            total_swaps += 1

            for g in front:
                gq1, gq2 = gates_2q[g]
                pair_key = (min(gq1, gq2), max(gq1, gq2))
                swap_counts[pair_key] += 1.0

        return swap_counts, total_swaps

    def simulate_routing_bidir(m, rm, max_layers=20):
        fwd_swaps, fwd_total = simulate_routing_greedy(m, rm, max_layers=max_layers)
        bwd_swaps, bwd_total = simulate_routing_backward(m, rm, max_layers=max_layers)
        combined = defaultdict(float)
        all_keys = set(fwd_swaps.keys()) | set(bwd_swaps.keys())
        for key in all_keys:
            combined[key] = 0.6 * fwd_swaps.get(key, 0.0) + 0.4 * bwd_swaps.get(key, 0.0)
        return combined, (fwd_total + bwd_total) // 2

    # ---------------------------------------------------------------
    # Step 8: Seed generation
    # ---------------------------------------------------------------
    sorted_by_degree = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    seed_lqs = sorted_by_degree[:min(3, len(sorted_by_degree))]
    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    logical_degree_ranked = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
    logical_degree_rank = {q: i for i, q in enumerate(logical_degree_ranked)}
    max_logical_rank = max(len(logical_degree_ranked) - 1, 1)
    phys_degree_ranked = sorted(physical_qubits, key=lambda pq: phys_degree[pq], reverse=True)
    phys_degree_rank = {pq: i for i, pq in enumerate(phys_degree_ranked)}
    max_phys_rank = max(len(phys_degree_ranked) - 1, 1)

    static_nbrs, static_deg = build_neighbors(static_weight)
    max_iw = max(static_weight.values()) if static_weight else 1.0

    # ---------------------------------------------------------------
    # Step 9: Greedy placement
    # ---------------------------------------------------------------
    def run_greedy_placement(start_lq, start_pq, nbrs, deg, weights=None):
        cur_max_iw = max(weights.values()) if weights else max_iw
        used_phys = {start_pq}
        m = [-1] * num_q
        rm = [-1] * num_q
        m[start_lq] = start_pq
        rm[start_pq] = start_lq
        placed = {start_lq}
        remaining = set(logical_qubits) - placed

        while remaining:
            best_lq, best_w = None, -1.0
            for lq in remaining:
                w = sum(nbrs.get(lq, {}).get(plq, 0.0) for plq in placed)
                if w > best_w:
                    best_w = w
                    best_lq = lq

            nbrs_placed = {plq: nbrs.get(best_lq, {}).get(plq, 0.0)
                           for plq in placed if plq in nbrs.get(best_lq, {})}

            if nbrs_placed:
                near_ties = []
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        d = dist[pq][m[plq]]
                        cost = iw * d
                        if m[plq] in hw_adj[pq]:
                            cost *= 0.90 - 0.10 * (iw / cur_max_iw)
                        score += cost
                    near_ties.append((score, pq))
                    if score < best_score:
                        best_score = score

                if near_ties and best_score > 0:
                    threshold = best_score * 1.05
                    candidates_list = [(s, pq) for s, pq in near_ties if s <= threshold]
                    if len(candidates_list) > 1 and best_lq in logical_degree_rank:
                        lq_rn = logical_degree_rank[best_lq] / max_logical_rank
                        best_pq = min(candidates_list, key=lambda x: (
                            abs(phys_degree_rank[x[1]] / max_phys_rank - lq_rn), x[0]))[1]
                    else:
                        best_pq = min(candidates_list, key=lambda x: x[0])[1]
                elif near_ties:
                    best_pq = min(near_ties, key=lambda x: x[0])[1]
                else:
                    best_pq = None
            else:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_phys:
                        score = phys_centrality[pq]
                        if score < best_score:
                            best_score = score
                            best_pq = pq

            m[best_lq] = best_pq
            rm[best_pq] = best_lq
            used_phys.add(best_pq)
            placed.add(best_lq)
            remaining.discard(best_lq)
        return m, rm

    # ---------------------------------------------------------------
    # Step 10: Spectral seed
    # ---------------------------------------------------------------
    def spectral_seed():
        if len(interacting_logical) < 3:
            return None, None

        idx_map = {q: i for i, q in enumerate(interacting_logical)}
        n = len(interacting_logical)

        L = [[0.0] * n for _ in range(n)]
        for (q1, q2), w in static_weight.items():
            if q1 in idx_map and q2 in idx_map:
                i, j = idx_map[q1], idx_map[q2]
                L[i][j] -= w
                L[j][i] -= w
                L[i][i] += w
                L[j][j] += w

        def power_iter(mat, n_iter=200):
            v = [random.gauss(0, 1) for _ in range(n)]
            for _ in range(n_iter):
                new_v = [0.0] * n
                for i in range(n):
                    for j in range(n):
                        new_v[i] += mat[i][j] * v[j]
                norm = math.sqrt(sum(x * x for x in new_v)) or 1e-12
                v = [x / norm for x in new_v]
            return v

        max_diag = max(L[i][i] for i in range(n)) + 1.0
        shifted = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                shifted[i][j] = -L[i][j]
            shifted[i][i] += max_diag

        v1 = power_iter(shifted, 150)
        dot_v1 = sum(x * x for x in v1) or 1e-12
        lam1_shifted = sum(v1[i] * sum(shifted[i][j] * v1[j] for j in range(n)) for i in range(n)) / dot_v1
        shifted2 = [row[:] for row in shifted]
        for i in range(n):
            for j in range(n):
                shifted2[i][j] -= lam1_shifted * v1[i] * v1[j] / dot_v1

        fiedler = power_iter(shifted2, 200)
        fiedler_order = sorted(range(n), key=lambda i: fiedler[i])
        sorted_logical = [interacting_logical[i] for i in fiedler_order]

        phys_sorted = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
        start_pq = phys_sorted[0]
        visited_bfs = set()
        bfs_order = []
        bfs_q = deque([start_pq])
        visited_bfs.add(start_pq)
        while bfs_q:
            pq = bfs_q.popleft()
            bfs_order.append(pq)
            for nb in sorted(hw_adj[pq]):
                if nb not in visited_bfs:
                    visited_bfs.add(nb)
                    bfs_q.append(nb)
        for pq in physical_qubits:
            if pq not in visited_bfs:
                bfs_order.append(pq)
                visited_bfs.add(pq)

        m = [-1] * num_q
        rm = [-1] * num_q
        n_interact = len(sorted_logical)
        n_phys = len(bfs_order)
        offset = max(0, (n_phys - n_interact) // 2)
        for i, lq in enumerate(sorted_logical):
            pq = bfs_order[min(offset + i, n_phys - 1)]
            if rm[pq] != -1:
                best_pq = None
                best_d = float('inf')
                for cpq in physical_qubits:
                    if rm[cpq] == -1:
                        d = dist[pq][cpq]
                        if d < best_d:
                            best_d = d
                            best_pq = cpq
                pq = best_pq
            m[lq] = pq
            rm[pq] = lq

        return m, rm

    # ---------------------------------------------------------------
    # Step 11: MIS seed
    # ---------------------------------------------------------------
    def conflict_graph_seed(start_pq, nbrs, deg, weights, routing_pressure=None):
        if len(interacting_logical) < 2:
            return None, None

        cur_max_iw = max(weights.values()) if weights else 1.0

        top_partners = {}
        for lq in interacting_logical:
            nbr_list = sorted(nbrs.get(lq, {}).items(), key=lambda x: x[1], reverse=True)
            top_partners[lq] = set(p for p, _ in nbr_list[:3])

        conflict_adj = defaultdict(set)
        il = interacting_logical

        qubit_pressure = defaultdict(float)
        if routing_pressure:
            for (q1, q2), cnt in routing_pressure.items():
                qubit_pressure[q1] += cnt
                qubit_pressure[q2] += cnt

        for i in range(len(il)):
            for j in range(i + 1, len(il)):
                l1, l2 = il[i], il[j]
                s1 = top_partners.get(l1, set())
                s2 = top_partners.get(l2, set())
                if not s1 and not s2:
                    continue
                intersection = len(s1 & s2)
                union = len(s1 | s2)
                if union > 0:
                    jaccard = intersection / union
                    threshold = 0.5
                    if routing_pressure and qubit_pressure[l1] > 0 and qubit_pressure[l2] > 0:
                        max_pressure = max(qubit_pressure.values()) if qubit_pressure else 1.0
                        pressure_factor = min(1.0, (qubit_pressure[l1] + qubit_pressure[l2]) /
                                              (max_pressure * 2 + 1e-10))
                        threshold = max(0.25, 0.5 - 0.25 * pressure_factor)
                    if jaccard > threshold:
                        conflict_adj[l1].add(l2)
                        conflict_adj[l2].add(l1)

        sorted_by_deg_local = sorted(interacting_logical,
                                     key=lambda q: (deg.get(q, 0), qubit_pressure.get(q, 0)),
                                     reverse=True)
        mis = set()
        excluded = set()
        for lq in sorted_by_deg_local:
            if lq not in excluded:
                mis.add(lq)
                for neighbor in conflict_adj.get(lq, set()):
                    excluded.add(neighbor)

        non_mis = [lq for lq in sorted_by_deg_local if lq not in mis]
        mis_ordered = sorted(mis, key=lambda q: (deg.get(q, 0), qubit_pressure.get(q, 0)),
                             reverse=True)

        m = [-1] * num_q
        rm = [-1] * num_q
        used_phys = set()

        if mis_ordered:
            first_lq = mis_ordered[0]
            m[first_lq] = start_pq
            rm[start_pq] = first_lq
            used_phys.add(start_pq)

            for lq in mis_ordered[1:]:
                nbrs_placed = {}
                for plq in mis:
                    if m[plq] >= 0 and plq in nbrs.get(lq, {}):
                        nbrs_placed[plq] = nbrs[lq][plq]

                if nbrs_placed:
                    best_pq = None
                    best_score = float('inf')
                    for pq in physical_qubits:
                        if pq in used_phys:
                            continue
                        score = 0.0
                        for plq, iw in nbrs_placed.items():
                            d = dist[pq][m[plq]]
                            cost = iw * d
                            if m[plq] in hw_adj[pq]:
                                cost *= 0.90 - 0.10 * (iw / cur_max_iw)
                            score += cost
                        if score < best_score:
                            best_score = score
                            best_pq = pq
                else:
                    best_pq = None
                    best_score = float('inf')
                    for pq in physical_qubits:
                        if pq not in used_phys:
                            if phys_centrality[pq] < best_score:
                                best_score = phys_centrality[pq]
                                best_pq = pq

                if best_pq is not None:
                    m[lq] = best_pq
                    rm[best_pq] = lq
                    used_phys.add(best_pq)

        placed_set = set(lq for lq in interacting_logical if m[lq] >= 0)

        for lq in non_mis:
            nbrs_placed = {}
            for plq in placed_set:
                w = nbrs.get(lq, {}).get(plq, 0.0)
                if w > 0:
                    nbrs_placed[plq] = w

            if nbrs_placed:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    score = 0.0
                    for plq, iw in nbrs_placed.items():
                        d = dist[pq][m[plq]]
                        cost = iw * d
                        if m[plq] in hw_adj[pq]:
                            cost *= 0.90 - 0.10 * (iw / cur_max_iw)
                        score += cost
                    if score < best_score:
                        best_score = score
                        best_pq = pq
            else:
                best_pq = None
                best_score = float('inf')
                for pq in physical_qubits:
                    if pq not in used_phys:
                        if phys_centrality[pq] < best_score:
                            best_score = phys_centrality[pq]
                            best_pq = pq

            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                used_phys.add(best_pq)
                placed_set.add(lq)

        remaining_lqs = [lq for lq in logical_qubits if m[lq] == -1]
        for lq in remaining_lqs:
            best_pq = None
            best_score = float('inf')
            for pq in physical_qubits:
                if pq not in used_phys:
                    if phys_centrality[pq] < best_score:
                        best_score = phys_centrality[pq]
                        best_pq = pq
            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                used_phys.add(best_pq)

        return m, rm

    # ---------------------------------------------------------------
    # Step 12: Local search
    # ---------------------------------------------------------------
    def local_search(m, rm, nbrs, weights, alpha_exp, max_rounds=5):
        if len(interacting_logical) <= 1:
            return compute_cost_nonlinear(m, weights, alpha_exp)
        for _ in range(max_rounds):
            improved = False
            best_d, best_pair = 0.0, None
            active_phys = [m[q] for q in interacting_logical]
            for pq1 in active_phys:
                for pq2 in hw_adj.get(pq1, set()):
                    d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                    if d < best_d:
                        best_d = d
                        best_pair = (pq1, pq2)
            n_random = min(150, len(interacting_logical) * 3)
            for _ in range(n_random):
                i, j = random.sample(range(len(interacting_logical)), 2)
                pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                if d < best_d:
                    best_d = d
                    best_pair = (pq1, pq2)
            if best_pair and best_d < -1e-12:
                do_swap(m, rm, *best_pair)
                improved = True
            if not improved:
                for i in range(len(interacting_logical)):
                    for j in range(i + 1, len(interacting_logical)):
                        pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                        d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                        if d < best_d:
                            best_d = d
                            best_pair = (pq1, pq2)
                if best_pair and best_d < -1e-12:
                    do_swap(m, rm, *best_pair)
                else:
                    break
        return compute_cost_nonlinear(m, weights, alpha_exp)

    # ---------------------------------------------------------------
    # Step 13: Perturbation modes with bandit selection
    # ---------------------------------------------------------------
    rsdiwr_iter_ref = [0]
    accumulated_routing_pressure_ref = [defaultdict(float)]

    def perturb_random(m, rm, **kw):
        if len(interacting_logical) >= 2:
            lqs = random.sample(interacting_logical, 2)
            do_swap(m, rm, m[lqs[0]], m[lqs[1]])

    def perturb_segment_shuffle(m, rm, **kw):
        k = min(random.randint(3, 5), len(interacting_logical))
        if k < 2:
            perturb_random(m, rm)
            return
        lqs = random.sample(interacting_logical, k)
        phys_positions = [m[lq] for lq in lqs]
        random.shuffle(phys_positions)
        for lq in lqs:
            rm[m[lq]] = -1
        for lq, pq in zip(lqs, phys_positions):
            m[lq] = pq
            rm[pq] = lq

    def perturb_worst_pair(m, rm, weights=None, alpha_exp=1.5, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        pair_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                d = dist[m[q1]][m[q2]]
                c = w * (d ** alpha_exp)
                pair_costs.append((c, q1, q2))
        if not pair_costs:
            return
        pair_costs.sort(reverse=True)
        top_n = min(3, len(pair_costs))
        _, tq1, tq2 = pair_costs[random.randint(0, top_n - 1)]
        pq1, pq2 = m[tq1], m[tq2]
        adj_of_pq2 = list(hw_adj.get(pq2, set()))
        if adj_of_pq2:
            target = random.choice(adj_of_pq2)
            if target != pq1:
                do_swap(m, rm, pq1, target)
            else:
                do_swap(m, rm, pq1, pq2)
        else:
            do_swap(m, rm, pq1, pq2)

    def perturb_edge_targeted(m, rm, weights=None, alpha_exp=1.5, **kw):
        if weights is None:
            perturb_random(m, rm)
            return
        edge_costs = []
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                c = w * (dist[m[q1]][m[q2]] ** alpha_exp)
                edge_costs.append((c, q1, q2))
        if not edge_costs:
            return
        edge_costs.sort(reverse=True)
        top_n = min(5, len(edge_costs))
        _, eq1, eq2 = edge_costs[random.randint(0, top_n - 1)]
        pq1, pq2 = m[eq1], m[eq2]
        neighbors_of_pq2 = list(hw_adj.get(pq2, set()))
        if neighbors_of_pq2:
            target = random.choice(neighbors_of_pq2)
            if target != pq1:
                do_swap(m, rm, pq1, target)
            else:
                do_swap(m, rm, pq1, pq2)
        else:
            do_swap(m, rm, pq1, pq2)

    def perturb_lns(m, rm, nbrs=None, alpha_exp=1.5, **kw):
        if nbrs is None or len(interacting_logical) < 3:
            perturb_random(m, rm)
            return
        base_lo = 3 + rsdiwr_iter_ref[0]
        base_hi = 8 + rsdiwr_iter_ref[0] * 2
        k = min(random.randint(base_lo, base_hi), len(interacting_logical))

        acc_pressure = accumulated_routing_pressure_ref[0]
        qubit_acc_pressure = defaultdict(float)
        for (q1, q2), cnt in acc_pressure.items():
            qubit_acc_pressure[q1] += cnt
            qubit_acc_pressure[q2] += cnt
        max_acc = max(qubit_acc_pressure.values()) if qubit_acc_pressure else 1.0

        qcost = {}
        for lq in interacting_logical:
            c = 0.0
            for partner, w in nbrs.get(lq, {}).items():
                if m[partner] >= 0 and m[lq] >= 0:
                    c += w * (dist[m[lq]][m[partner]] ** alpha_exp)
            pressure_bonus = qubit_acc_pressure.get(lq, 0.0) / (max_acc + 1e-10)
            qcost[lq] = c * (1.0 + 0.5 * pressure_bonus)

        sorted_qs = sorted(qcost, key=lambda q: qcost[q], reverse=True)
        top_half = sorted_qs[:max(k, len(sorted_qs) // 2)]
        subset = random.sample(top_half, min(k, len(top_half)))

        freed_phys = []
        for lq in subset:
            freed_phys.append(m[lq])
            rm[m[lq]] = -1
            m[lq] = -1

        placed_set = set(lq for lq in interacting_logical if m[lq] >= 0)
        for lq in sorted(subset, key=lambda q: qcost[q], reverse=True):
            best_pq, best_sc = None, float('inf')
            for pq in freed_phys:
                if rm[pq] != -1:
                    continue
                sc = 0.0
                for partner, w in nbrs.get(lq, {}).items():
                    if partner in placed_set and m[partner] >= 0:
                        sc += w * (dist[pq][m[partner]] ** alpha_exp)
                if sc < best_sc:
                    best_sc = sc
                    best_pq = pq
            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                placed_set.add(lq)
            else:
                for pq in freed_phys:
                    if rm[pq] == -1:
                        m[lq] = pq
                        rm[pq] = lq
                        placed_set.add(lq)
                        break

    perturbation_modes = [perturb_random, perturb_segment_shuffle,
                          perturb_worst_pair, perturb_edge_targeted, perturb_lns]
    K = len(perturbation_modes)

    WINDOW_SIZE = 100
    EPSILON = 0.1
    window = []
    mode_successes = [0] * K
    mode_attempts = [0] * K

    def select_perturbation_mode():
        if random.random() < EPSILON or sum(mode_attempts) < K * 2:
            return random.randint(0, K - 1)
        rates = []
        for k in range(K):
            if mode_attempts[k] > 0:
                rates.append(mode_successes[k] / mode_attempts[k])
            else:
                rates.append(1.0)
        total = sum(rates)
        if total < 1e-12:
            return random.randint(0, K - 1)
        r = random.random() * total
        cumul = 0.0
        for k in range(K):
            cumul += rates[k]
            if r <= cumul:
                return k
        return K - 1

    def update_window(mode_idx, success):
        window.append((mode_idx, success))
        mode_attempts[mode_idx] += 1
        if success:
            mode_successes[mode_idx] += 1
        if len(window) > WINDOW_SIZE:
            old_mode, old_success = window.pop(0)
            mode_attempts[old_mode] -= 1
            if old_success:
                mode_successes[old_mode] -= 1

    # ---------------------------------------------------------------
    # Step 14: RSDIWR+ILS engine (reusable at any resolution)
    # ---------------------------------------------------------------
    def run_rsdiwr_ils(init_m, init_rm, n_rsdiwr_iters, time_budget_sec, t_start_ref):
        """Run RSDIWR+ILS from given initial mapping. Returns best_m, best_rm, best_cost."""
        cur_m = list(init_m)
        cur_rm = list(init_rm)
        best_m = list(init_m)
        best_rm = list(init_rm)
        best_cost = compute_cost(init_m, static_weight)
        swap_counts = defaultdict(float)
        accumulated_routing_pressure = defaultdict(float)
        accumulated_routing_pressure_ref[0] = accumulated_routing_pressure

        for rsdiwr_iter in range(n_rsdiwr_iters):
            rsdiwr_iter_ref[0] = rsdiwr_iter
            elapsed = _time() - t_start_ref
            if elapsed > time_budget_sec:
                break

            remaining_time = time_budget_sec - elapsed
            remaining_iters = n_rsdiwr_iters - rsdiwr_iter
            time_for_this_iter = remaining_time / max(remaining_iters, 1)
            n_ils = max(55, int(num_q * time_for_this_iter / 0.5)) if time_for_this_iter > 1.0 else max(30, num_q // 3)

            if rsdiwr_iter == 0 or not swap_counts:
                eff_weights = dict(static_weight)
            else:
                max_sw = max(swap_counts.values()) if swap_counts else 1.0
                scale = max(static_weight.values()) / max(max_sw, 1e-10)
                total_pressure = sum(accumulated_routing_pressure.values())
                n_pairs = len(accumulated_routing_pressure)
                avg_pressure = total_pressure / max(n_pairs, 1)
                pressure_adapt = min(0.15, 0.05 * avg_pressure / max(max_sw, 1e-10))
                alpha_blend = max(0.25, 1.0 - 0.3 * rsdiwr_iter - pressure_adapt)

                eff_weights = defaultdict(float)
                all_keys = set(static_weight.keys()) | set(swap_counts.keys())
                for key in all_keys:
                    w_s = static_weight.get(key, 0.0)
                    w_r = swap_counts.get(key, 0.0) * scale
                    eff_weights[key] = alpha_blend * w_s + (1.0 - alpha_blend) * w_r

            eff_nbrs, eff_deg = build_neighbors(eff_weights)

            # Re-seed with routing-informed weights
            if rsdiwr_iter > 0 and swap_counts and (_time() - t_start_ref) < time_budget_sec * 0.7:
                eff_sorted = sorted(interacting_logical,
                                    key=lambda q: eff_deg.get(q, 0), reverse=True)
                reseed_lqs = eff_sorted[:min(2, len(eff_sorted))]
                reseed_pqs = seed_pqs[:min(2, len(seed_pqs))]

                for s_lq in reseed_lqs:
                    for s_pq in reseed_pqs:
                        rm_new, rmm_new = run_greedy_placement(
                            s_lq, s_pq, eff_nbrs, eff_deg, weights=eff_weights)
                        fill_unmapped(rm_new, rmm_new)
                        local_search(rm_new, rmm_new, eff_nbrs, eff_weights,
                                     alpha_exp=2.0, max_rounds=3)
                        sc = compute_cost(rm_new, static_weight)
                        if sc < best_cost:
                            best_cost = sc
                            best_m = list(rm_new)
                            best_rm = list(rmm_new)
                            cur_m = list(rm_new)
                            cur_rm = list(rmm_new)

                for s_pq in reseed_pqs:
                    cg_m, cg_rm = conflict_graph_seed(
                        s_pq, eff_nbrs, eff_deg, eff_weights,
                        routing_pressure=accumulated_routing_pressure)
                    if cg_m is not None:
                        fill_unmapped(cg_m, cg_rm)
                        local_search(cg_m, cg_rm, eff_nbrs, eff_weights,
                                     alpha_exp=2.0, max_rounds=3)
                        sc = compute_cost(cg_m, static_weight)
                        if sc < best_cost:
                            best_cost = sc
                            best_m = list(cg_m)
                            best_rm = list(cg_rm)
                            cur_m = list(cg_m)
                            cur_rm = list(cg_rm)

            alpha_start = 2.0
            alpha_end = 1.0

            ls_rounds = 6 if rsdiwr_iter == 0 else 4
            cur_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights,
                                       alpha_exp=alpha_start, max_rounds=ls_rounds)

            static_cost = compute_cost(cur_m, static_weight)
            if static_cost < best_cost:
                best_cost = static_cost
                best_m = list(cur_m)
                best_rm = list(cur_rm)

            # Reset bandit window
            window.clear()
            for k in range(K):
                mode_successes[k] = 0
                mode_attempts[k] = 0

            T = max(cur_cost_nl * 0.05, 1.0)
            T_init = T
            alpha_sa = 0.94
            reheat_interval = max(n_ils // 4, 8)

            for ils_iter in range(n_ils):
                if _time() - t_start_ref > time_budget_sec:
                    break

                progress = ils_iter / max(n_ils - 1, 1)
                alpha_exp = alpha_start + (alpha_end - alpha_start) * progress

                saved_m = list(cur_m)
                saved_rm = list(cur_rm)
                saved_cost = cur_cost_nl

                mode = select_perturbation_mode()
                perturbation_modes[mode](cur_m, cur_rm,
                                        nbrs=eff_nbrs, weights=eff_weights,
                                        alpha_exp=alpha_exp)

                new_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights,
                                           alpha_exp=alpha_exp, max_rounds=3)

                success = new_cost_nl < saved_cost - 1e-12

                improvement = saved_cost - new_cost_nl
                if improvement > 0:
                    cur_cost_nl = new_cost_nl
                    sc = compute_cost(cur_m, static_weight)
                    if sc < best_cost:
                        best_cost = sc
                        best_m = list(cur_m)
                        best_rm = list(cur_rm)
                elif random.random() < math.exp(min(0, improvement / max(T, 1e-10))):
                    cur_cost_nl = new_cost_nl
                else:
                    cur_m[:] = saved_m
                    cur_rm[:] = saved_rm
                    cur_cost_nl = saved_cost

                update_window(mode, success)

                T *= alpha_sa
                if (ils_iter + 1) % reheat_interval == 0:
                    T = max(T, T_init * 0.35)

            # Routing simulation for RSDIWR feedback
            if rsdiwr_iter < n_rsdiwr_iters - 1:
                sim_depth = 14 + rsdiwr_iter * 7
                swap_counts, _ = simulate_routing_bidir(best_m, best_rm, max_layers=sim_depth)

                for pair_key, cnt in swap_counts.items():
                    accumulated_routing_pressure[pair_key] += cnt

            cur_m = list(best_m)
            cur_rm = list(best_rm)
            cur_cost_nl = compute_cost_nonlinear(cur_m, eff_weights, alpha_end)

        return best_m, best_rm, best_cost

    # ===============================================================
    # PHASE 1: COARSE-LEVEL CONTRACTION AND RSDIWR
    # ===============================================================
    t_global_start = _time()
    TOTAL_TIME_BUDGET = 25.0

    # Determine contraction pairs using maximum-weight matching
    # on edges above 90th percentile
    interacting_set = set(interacting_logical)

    if len(interacting_logical) >= 6 and len(static_weight) >= 3:
        # Compute 90th percentile threshold for contraction
        weight_vals = sorted(static_weight.values())
        p90_idx = int(len(weight_vals) * 0.90)
        p90_threshold = weight_vals[min(p90_idx, len(weight_vals) - 1)]

        # Filter to high-weight edges for contraction matching
        high_weight_edges = {k: v for k, v in static_weight.items() if v >= p90_threshold}

        # Maximum-weight matching on interaction graph (greedy)
        logical_pairs = greedy_max_weight_matching(high_weight_edges, interacting_set)

        # Contract interaction graph
        contracted_weights, super_to_pair, qubit_to_super = contract_interaction_graph(
            static_weight, logical_pairs, interacting_set)

        # Contract hardware graph: match adjacent physical pairs with highest combined degree
        hw_pairs = greedy_max_weight_matching_hw(hw_adj, set(physical_qubits))

        contracted_dist, super_to_phys, phys_to_super, contracted_hw_adj, n_super_phys = contract_hardware_graph(
            hw_pairs, physical_qubits)

        # Number of super-qubits (logical) and super-positions (physical)
        n_super_logical = len(super_to_pair)

        # Only proceed with coarse phase if we actually contracted significantly
        if n_super_logical < len(interacting_logical) * 0.85 and n_super_phys < len(physical_qubits) * 0.85:

            # Build coarse-level greedy placement
            # Map super-qubit IDs to their total interaction degree
            super_degree = defaultdict(float)
            for (s1, s2), w in contracted_weights.items():
                super_degree[s1] += w
                super_degree[s2] += w

            super_nodes = sorted(super_to_pair.keys())
            super_phys_nodes = sorted(super_to_phys.keys())

            # Coarse centrality
            coarse_centrality = {}
            for sp in super_phys_nodes:
                coarse_centrality[sp] = sum(contracted_dist[sp][sp2] for sp2 in super_phys_nodes)

            # Greedy placement at coarse level
            def coarse_greedy_placement(start_super_lq, start_super_pq):
                cm = [-1] * max(max(super_nodes) + 1, n_super_phys)
                crm = [-1] * max(max(super_nodes) + 1, n_super_phys)
                # Extend arrays if needed
                while len(cm) <= max(max(super_nodes, default=0), max(super_phys_nodes, default=0)):
                    cm.append(-1)
                    crm.append(-1)
                sz = max(len(cm), len(crm), max(super_nodes, default=0) + 1, max(super_phys_nodes, default=0) + 1)
                cm = [-1] * sz
                crm = [-1] * sz

                cm[start_super_lq] = start_super_pq
                crm[start_super_pq] = start_super_lq
                placed = {start_super_lq}
                used = {start_super_pq}
                remaining = set(super_nodes) - placed

                coarse_nbrs = defaultdict(dict)
                for (s1, s2), w in contracted_weights.items():
                    coarse_nbrs[s1][s2] = w
                    coarse_nbrs[s2][s1] = w

                while remaining:
                    best_sq, best_w = None, -1.0
                    for sq in remaining:
                        w = sum(coarse_nbrs.get(sq, {}).get(psq, 0.0) for psq in placed)
                        if w > best_w:
                            best_w = w
                            best_sq = sq

                    # Find best physical super-position
                    best_sp, best_score = None, float('inf')
                    for sp in super_phys_nodes:
                        if sp in used:
                            continue
                        score = 0.0
                        for psq in placed:
                            iw = coarse_nbrs.get(best_sq, {}).get(psq, 0.0)
                            if iw > 0:
                                score += iw * contracted_dist[sp][cm[psq]]
                        if score < best_score:
                            best_score = score
                            best_sp = sp

                    if best_sp is None:
                        # Use any free position
                        for sp in super_phys_nodes:
                            if sp not in used:
                                best_sp = sp
                                break

                    if best_sp is not None:
                        cm[best_sq] = best_sp
                        crm[best_sp] = best_sq
                        placed.add(best_sq)
                        used.add(best_sp)
                    remaining.discard(best_sq)

                return cm, crm

            # Generate a few coarse seeds
            sorted_super_by_deg = sorted(super_nodes, key=lambda s: super_degree.get(s, 0), reverse=True)
            sorted_super_phys_by_cent = sorted(super_phys_nodes, key=lambda s: coarse_centrality.get(s, 0))

            best_coarse_cost = float('inf')
            best_coarse_cm = None

            for s_lq in sorted_super_by_deg[:min(2, len(sorted_super_by_deg))]:
                for s_pq in sorted_super_phys_by_cent[:min(2, len(sorted_super_phys_by_cent))]:
                    cm, crm = coarse_greedy_placement(s_lq, s_pq)
                    cost = 0.0
                    for (s1, s2), w in contracted_weights.items():
                        if cm[s1] >= 0 and cm[s2] >= 0:
                            cost += w * contracted_dist[cm[s1]][cm[s2]]
                    if cost < best_coarse_cost:
                        best_coarse_cost = cost
                        best_coarse_cm = (list(cm), list(crm))

            # ===============================================================
            # PHASE 2: EXPAND COARSE SOLUTION TO FULL RESOLUTION
            # ===============================================================
            if best_coarse_cm is not None:
                coarse_m, coarse_rm = best_coarse_cm

                # Build full-resolution mapping from coarse solution
                expanded_m = [-1] * num_q
                expanded_rm = [-1] * num_q
                used_phys_expanded = set()

                for super_lq in super_nodes:
                    super_pq = coarse_m[super_lq]
                    if super_pq < 0:
                        continue

                    logical_members = super_to_pair[super_lq]
                    physical_members = super_to_phys.get(super_pq, ())

                    if len(logical_members) == 1:
                        # Singleton: assign to first available physical member
                        lq = logical_members[0]
                        for pq in physical_members:
                            if pq not in used_phys_expanded:
                                expanded_m[lq] = pq
                                expanded_rm[pq] = lq
                                used_phys_expanded.add(pq)
                                break
                    elif len(logical_members) == 2:
                        lq1, lq2 = logical_members
                        # Assign higher-interaction qubit to physical qubit with more free neighbors
                        d1 = logical_degree.get(lq1, 0)
                        d2 = logical_degree.get(lq2, 0)
                        if d1 < d2:
                            lq1, lq2 = lq2, lq1  # lq1 has higher interaction

                        if len(physical_members) >= 2:
                            p1, p2 = physical_members[0], physical_members[1]
                            # Count free neighbors for each physical position
                            free_nb_p1 = sum(1 for nb in hw_adj.get(p1, set()) if nb not in used_phys_expanded)
                            free_nb_p2 = sum(1 for nb in hw_adj.get(p2, set()) if nb not in used_phys_expanded)

                            if free_nb_p1 >= free_nb_p2:
                                if p1 not in used_phys_expanded:
                                    expanded_m[lq1] = p1
                                    expanded_rm[p1] = lq1
                                    used_phys_expanded.add(p1)
                                if p2 not in used_phys_expanded:
                                    expanded_m[lq2] = p2
                                    expanded_rm[p2] = lq2
                                    used_phys_expanded.add(p2)
                            else:
                                if p2 not in used_phys_expanded:
                                    expanded_m[lq1] = p2
                                    expanded_rm[p2] = lq1
                                    used_phys_expanded.add(p2)
                                if p1 not in used_phys_expanded:
                                    expanded_m[lq2] = p1
                                    expanded_rm[p1] = lq2
                                    used_phys_expanded.add(p1)
                        elif len(physical_members) == 1:
                            pq = physical_members[0]
                            if pq not in used_phys_expanded:
                                expanded_m[lq1] = pq
                                expanded_rm[pq] = lq1
                                used_phys_expanded.add(pq)

                # Place any still-unplaced logical qubits (could happen if expansion incomplete)
                unplaced = [lq for lq in logical_qubits if expanded_m[lq] == -1]
                for lq in sorted(unplaced, key=lambda q: logical_degree.get(q, 0), reverse=True):
                    # Find closest free physical qubit to qubits it interacts with
                    best_pq, best_score = None, float('inf')
                    nbrs_placed = {}
                    for partner, w in static_nbrs.get(lq, {}).items():
                        if expanded_m[partner] >= 0:
                            nbrs_placed[partner] = w

                    if nbrs_placed:
                        for pq in physical_qubits:
                            if pq in used_phys_expanded:
                                continue
                            score = 0.0
                            for partner, iw in nbrs_placed.items():
                                score += iw * dist[pq][expanded_m[partner]]
                            if score < best_score:
                                best_score = score
                                best_pq = pq
                    else:
                        for pq in physical_qubits:
                            if pq not in used_phys_expanded:
                                if phys_centrality[pq] < best_score:
                                    best_score = phys_centrality[pq]
                                    best_pq = pq

                    if best_pq is not None:
                        expanded_m[lq] = best_pq
                        expanded_rm[best_pq] = lq
                        used_phys_expanded.add(best_pq)

                fill_unmapped(expanded_m, expanded_rm)

                coarse_expanded_seed = (expanded_m, expanded_rm)
            else:
                coarse_expanded_seed = None
        else:
            coarse_expanded_seed = None
    else:
        coarse_expanded_seed = None

    # ===============================================================
    # PHASE 3: FINE-RESOLUTION RSDIWR+ILS
    # ===============================================================
    # Build initial candidates from multiple seeds
    candidates = []

    # Seed from coarse expansion
    if coarse_expanded_seed is not None:
        cm, crm = coarse_expanded_seed
        cost = compute_cost(cm, static_weight)
        candidates.append((cost, list(cm), list(crm)))

    # Standard greedy seeds
    for s_lq in seed_lqs:
        for s_pq in seed_pqs:
            m, rm = run_greedy_placement(s_lq, s_pq, static_nbrs, static_deg,
                                         weights=static_weight)
            fill_unmapped(m, rm)
            cost = compute_cost(m, static_weight)
            candidates.append((cost, m, rm))

    # MIS seeds
    for s_pq in seed_pqs:
        cg_m, cg_rm = conflict_graph_seed(s_pq, static_nbrs, static_deg, static_weight)
        if cg_m is not None:
            fill_unmapped(cg_m, cg_rm)
            cost = compute_cost(cg_m, static_weight)
            candidates.append((cost, cg_m, cg_rm))

    # Spectral seed
    spec_m, spec_rm = spectral_seed()
    if spec_m is not None:
        fill_unmapped(spec_m, spec_rm)
        spec_cost = compute_cost(spec_m, static_weight)
        candidates.append((spec_cost, spec_m, spec_rm))

    # Random seed
    m_rand = [-1] * num_q
    rm_rand = [-1] * num_q
    shuffled_phys = list(physical_qubits)
    random.shuffle(shuffled_phys)
    for i, lq in enumerate(logical_qubits):
        m_rand[lq] = shuffled_phys[i]
        rm_rand[shuffled_phys[i]] = lq
    fill_unmapped(m_rand, rm_rand)
    c = compute_cost(m_rand, static_weight)
    candidates.append((c, m_rand, rm_rand))

    if not candidates:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    candidates.sort(key=lambda x: x[0])

    # Local search on top candidates
    _, best_m, best_rm = candidates[0]
    best_m, best_rm = list(best_m), list(best_rm)
    local_search(best_m, best_rm, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=6)
    best_cost = compute_cost(best_m, static_weight)

    for idx in range(1, min(6, len(candidates))):
        _, m_c, rm_c = candidates[idx]
        m_c, rm_c = list(m_c), list(rm_c)
        local_search(m_c, rm_c, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=4)
        c = compute_cost(m_c, static_weight)
        if c < best_cost:
            best_cost = c
            best_m = list(m_c)
            best_rm = list(rm_c)

    # Run RSDIWR+ILS from best seed
    elapsed_so_far = _time() - t_global_start
    remaining_budget = TOTAL_TIME_BUDGET - elapsed_so_far

    if remaining_budget > 1.0:
        best_m, best_rm, best_cost = run_rsdiwr_ils(
            best_m, best_rm, n_rsdiwr_iters=4,
            time_budget_sec=TOTAL_TIME_BUDGET,
            t_start_ref=t_global_start)

    # ---------------------------------------------------------------
    # Final: Set mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)