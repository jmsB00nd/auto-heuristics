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
    # Step 2: Commutativity-aware structures
    # Build adjacency in topo order and commutativity checker
    # Two CX gates commute iff they act on disjoint qubit sets
    # ---------------------------------------------------------------
    def gates_commute(g1, g2):
        """Check if two 2-qubit gates commute (disjoint qubit sets)."""
        q1a, q1b = gates_2q[g1]
        q2a, q2b = gates_2q[g2]
        return q1a != q2a and q1a != q2b and q1b != q2a and q1b != q2b

    def recompute_topo_and_deps():
        """Recompute topological order and dependency counts after DAG relinearization."""
        nonlocal topo_2q, dep_count, bwd_dep
        new_in_deg = {g: len(dag2q_pred[g]) for g in gates_2q}
        new_topo = []
        qq = deque(sorted(g for g in gates_2q if new_in_deg[g] == 0))
        while qq:
            g = qq.popleft()
            new_topo.append(g)
            for s in dag2q_succ[g]:
                new_in_deg[s] -= 1
                if new_in_deg[s] == 0:
                    qq.append(s)
        topo_2q = new_topo

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
    # Step 3: Interaction weights — critical-path weighting
    # ---------------------------------------------------------------
    def compute_interaction_weights():
        max_layer_val = max((gate_layer[g] for g in two_qubit_gates), default=1)
        sw = defaultdict(float)
        ld = defaultdict(float)
        for g in two_qubit_gates:
            q1, q2 = self.access[g]
            cp = critical_path[g] + 1
            layer = gate_layer[g]
            w = cp * (max_layer_val - layer + 1)
            key = (min(q1, q2), max(q1, q2))
            sw[key] += w
            ld[q1] += w
            ld[q2] += w
        return sw, ld

    static_weight, logical_degree = compute_interaction_weights()
    interacting_logical = [q for q in logical_qubits if logical_degree.get(q, 0) > 0]

    # ---------------------------------------------------------------
    # Step 4: Physical graph properties
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
    # Step 5: Helper functions
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

    # ---------------------------------------------------------------
    # Step 6: Forward greedy routing simulation with bottleneck tracking
    # ---------------------------------------------------------------
    def simulate_routing_greedy(m, rm, max_layers=20, track_bottlenecks=False):
        sim_m = list(m)
        sim_rm = list(rm)
        swap_counts = defaultdict(float)
        total_swaps = 0
        gate_swap_cost = defaultdict(int)  # per-gate swap cost for bottleneck detection

        if not gates_2q:
            return swap_counts, total_swaps, gate_swap_cost

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
                if track_bottlenecks:
                    gate_swap_cost[g] += 1

        return swap_counts, total_swaps, gate_swap_cost

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

    def simulate_routing_asymmetric_bidir(m, rm, max_layers=20, track_bottlenecks=False):
        fwd_swaps, fwd_total, fwd_bottlenecks = simulate_routing_greedy(m, rm, max_layers=max_layers, track_bottlenecks=track_bottlenecks)
        bwd_swaps, bwd_total = simulate_routing_backward(m, rm, max_layers=max_layers)

        combined = defaultdict(float)
        all_keys = set(fwd_swaps.keys()) | set(bwd_swaps.keys())
        for key in all_keys:
            combined[key] = 0.6 * fwd_swaps.get(key, 0.0) + 0.4 * bwd_swaps.get(key, 0.0)

        return combined, (fwd_total + bwd_total) // 2, fwd_bottlenecks

    # ---------------------------------------------------------------
    # Step 7: DAG Relinearization — the key novelty
    # Swap adjacent commuting gates in topo order when bottleneck
    # gates could benefit from different front-layer composition
    # ---------------------------------------------------------------
    def relinearize_dag(gate_swap_cost, m, rm, max_swaps=20):
        """Reorder commuting adjacent gates in the DAG to reduce bottlenecks."""
        if not gate_swap_cost:
            return False

        # Identify bottleneck gates (needed >=2 SWAPs)
        bottleneck_gates = set(g for g, cost in gate_swap_cost.items() if cost >= 2)
        if not bottleneck_gates:
            return False

        # Build topo position index
        topo_pos = {g: i for i, g in enumerate(topo_2q)}

        # Find commuting adjacent pairs involving bottleneck gates
        # Adjacent = one is direct predecessor of the other in 2q DAG
        swaps_done = 0
        changed = False

        for bn_gate in sorted(bottleneck_gates, key=lambda g: gate_swap_cost.get(g, 0), reverse=True):
            if swaps_done >= max_swaps:
                break

            # Try swapping with predecessors
            for pred_gate in list(dag2q_pred.get(bn_gate, set())):
                if pred_gate not in gates_2q:
                    continue
                if swaps_done >= max_swaps:
                    break

                # Check commutativity (disjoint qubit sets)
                if not gates_commute(bn_gate, pred_gate):
                    continue

                # Check that pred_gate is the ONLY connection between them
                # i.e., we can safely swap their order
                # pred_gate -> bn_gate edge exists; check if removing it
                # and adding bn_gate -> pred_gate maintains a valid DAG

                # Verify: bn_gate has no other dependency on pred_gate through other paths
                # and pred_gate has no dependency on bn_gate through other paths
                # Since they commute (disjoint qubits), swapping is always safe
                # as long as it doesn't create a cycle

                # Check no other path from pred_gate to bn_gate
                other_preds_of_bn = dag2q_pred[bn_gate] - {pred_gate}
                other_succs_of_pred = dag2q_succ[pred_gate] - {bn_gate}

                # Simple cycle check: ensure bn_gate is not an ancestor of pred_gate
                # through any other path
                safe = True
                # BFS from bn_gate through other successors to see if we reach pred_gate
                visited = set()
                bfs_q = deque()
                for s in dag2q_succ.get(bn_gate, set()):
                    if s != pred_gate:
                        bfs_q.append(s)
                        visited.add(s)
                while bfs_q:
                    node = bfs_q.popleft()
                    if node == pred_gate:
                        safe = False
                        break
                    for s in dag2q_succ.get(node, set()):
                        if s not in visited:
                            visited.add(s)
                            bfs_q.append(s)

                if not safe:
                    continue

                # Perform the swap: remove pred->bn edge, add bn->pred edge
                dag2q_succ[pred_gate].discard(bn_gate)
                dag2q_pred[bn_gate].discard(pred_gate)
                dag2q_succ[bn_gate].add(pred_gate)
                dag2q_pred[pred_gate].add(bn_gate)

                swaps_done += 1
                changed = True

            # Try swapping with successors
            for succ_gate in list(dag2q_succ.get(bn_gate, set())):
                if succ_gate not in gates_2q:
                    continue
                if swaps_done >= max_swaps:
                    break

                if not gates_commute(bn_gate, succ_gate):
                    continue

                # Check no cycle: ensure succ_gate is not an ancestor of bn_gate
                # through other paths
                safe = True
                visited = set()
                bfs_q = deque()
                for p in dag2q_pred.get(succ_gate, set()):
                    if p != bn_gate and p in gates_2q:
                        bfs_q.append(p)
                        visited.add(p)
                while bfs_q:
                    node = bfs_q.popleft()
                    if node == bn_gate:
                        safe = False
                        break
                    for p in dag2q_pred.get(node, set()):
                        if p in gates_2q and p not in visited:
                            visited.add(p)
                            bfs_q.append(p)

                if not safe:
                    continue

                # Swap: remove bn->succ edge, add succ->bn edge
                dag2q_succ[bn_gate].discard(succ_gate)
                dag2q_pred[succ_gate].discard(bn_gate)
                dag2q_succ[succ_gate].add(bn_gate)
                dag2q_pred[bn_gate].add(succ_gate)

                swaps_done += 1
                changed = True

        if changed:
            recompute_topo_and_deps()

        return changed

    # ---------------------------------------------------------------
    # Step 8: A*-guided early-layer routing
    # ---------------------------------------------------------------
    ASTAR_LAYERS = 5
    n_2q = len(gates_2q)
    if n_2q < 50:
        ASTAR_NODE_LIMIT = 8000
    elif n_2q < 200:
        ASTAR_NODE_LIMIT = 5000
    else:
        ASTAR_NODE_LIMIT = 3000

    pred_rem_layer = {g: len(dag2q_pred[g]) for g in gates_2q}
    front_layer_calc = set(g for g in gates_2q if pred_rem_layer[g] == 0)
    current_layer = 0
    early_layer_gates = set()

    temp_front = set(front_layer_calc)
    while temp_front:
        for g in temp_front:
            if current_layer < ASTAR_LAYERS:
                early_layer_gates.add(g)
        next_front = set()
        for g in temp_front:
            for s in dag2q_succ[g]:
                pred_rem_layer[s] -= 1
                if pred_rem_layer[s] == 0:
                    next_front.add(s)
        current_layer += 1
        temp_front = next_front

    def astar_routing_early(m, rm):
        if not early_layer_gates:
            return defaultdict(float), 0

        early_pred = {}
        for g in early_layer_gates:
            early_pred[g] = len(dag2q_pred[g] & early_layer_gates)

        initial_front = frozenset(g for g in early_layer_gates if early_pred[g] == 0)

        init_executed = set()
        cur_front = set(initial_front)
        changed_flag = True
        while changed_flag:
            changed_flag = False
            newly_exec = set()
            for g in cur_front:
                gq1, gq2 = gates_2q[g]
                p1, p2 = m[gq1], m[gq2]
                if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
                    newly_exec.add(g)
            if newly_exec:
                changed_flag = True
                init_executed |= newly_exec
                cur_front -= newly_exec
                for g in newly_exec:
                    for s in dag2q_succ[g]:
                        if s in early_layer_gates and s not in init_executed:
                            all_pred_done = True
                            for p in dag2q_pred[s]:
                                if p in early_layer_gates and p not in init_executed:
                                    all_pred_done = False
                                    break
                            if all_pred_done:
                                cur_front.add(s)

        remaining_early = early_layer_gates - init_executed
        if not remaining_early:
            return defaultdict(float), 0

        def compute_front(executed):
            front = set()
            for g in early_layer_gates:
                if g in executed:
                    continue
                all_pred = True
                for p in dag2q_pred[g]:
                    if p in early_layer_gates and p not in executed:
                        all_pred = False
                        break
                if all_pred:
                    front.add(g)
            return frozenset(front)

        init_exec_frozen = frozenset(init_executed)
        init_front_astar = compute_front(init_executed)

        active_logical = set()
        for g in remaining_early:
            gq1, gq2 = gates_2q[g]
            active_logical.add(gq1)
            active_logical.add(gq2)

        def h_func(mapping, front_gates):
            h = 0
            for g in front_gates:
                gq1, gq2 = gates_2q[g]
                h += dist[mapping[gq1]][mapping[gq2]]
            return h

        active_logical_list = sorted(active_logical)

        def mapping_key(mapping):
            return tuple(mapping[lq] for lq in active_logical_list)

        init_m = list(m)
        init_h = h_func(init_m, init_front_astar)

        counter = [0]
        start_state = (init_h, 0, 0, init_m, init_exec_frozen, init_front_astar)
        open_heap = [start_state]
        visited = {}
        best_known_total = float('inf')
        best_result = None

        swap_history = {}
        init_key = (mapping_key(init_m), init_exec_frozen)
        swap_history[init_key] = defaultdict(float)

        expansions = 0

        while open_heap and expansions < ASTAR_NODE_LIMIT:
            f, _, g_cost, cur_m_list, executed, front = heapq.heappop(open_heap)

            mk = mapping_key(cur_m_list)
            state_key = (mk, executed)

            if state_key in visited and visited[state_key] <= g_cost:
                continue
            visited[state_key] = g_cost

            if best_known_total < float('inf') and f > best_known_total * 1.2:
                continue

            new_executed = set(executed)
            new_front = set(front)
            progress = True
            while progress:
                progress = False
                newly_exec = set()
                for g in list(new_front):
                    gq1, gq2 = gates_2q[g]
                    p1, p2 = cur_m_list[gq1], cur_m_list[gq2]
                    if (p1, p2) in self.backend_connections or (p2, p1) in self.backend_connections:
                        newly_exec.add(g)
                if newly_exec:
                    progress = True
                    new_executed |= newly_exec
                    new_front -= newly_exec
                    for g in newly_exec:
                        for s in dag2q_succ[g]:
                            if s in early_layer_gates and s not in new_executed:
                                all_pred = True
                                for p in dag2q_pred[s]:
                                    if p in early_layer_gates and p not in new_executed:
                                        all_pred = False
                                        break
                                if all_pred:
                                    new_front.add(s)

            new_exec_frozen = frozenset(new_executed)
            new_front_frozen = frozenset(new_front)

            if not (early_layer_gates - new_executed):
                if g_cost < best_known_total:
                    best_known_total = g_cost
                    parent_key = (mk, executed)
                    pair_swaps = dict(swap_history.get(parent_key, {}))
                    best_result = (g_cost, pair_swaps)
                continue

            if not new_front_frozen:
                continue

            active_phys = set()
            for g in new_front_frozen:
                gq1, gq2 = gates_2q[g]
                active_phys.add(cur_m_list[gq1])
                active_phys.add(cur_m_list[gq2])
            ext_gates = set()
            for g in new_front_frozen:
                for s in dag2q_succ[g]:
                    if s in early_layer_gates and s not in new_executed:
                        ext_gates.add(s)
            for g in ext_gates:
                gq1, gq2 = gates_2q[g]
                active_phys.add(cur_m_list[gq1])
                active_phys.add(cur_m_list[gq2])

            swaps = set()
            for pq in active_phys:
                for nb in hw_adj.get(pq, set()):
                    swaps.add((min(pq, nb), max(pq, nb)))

            expansions += 1
            parent_pair_swaps = swap_history.get((mk, executed), defaultdict(float))

            for (s1, s2) in swaps:
                new_m = list(cur_m_list)
                lq_at_s1 = -1
                lq_at_s2 = -1
                for lq in range(num_q):
                    if new_m[lq] == s1:
                        lq_at_s1 = lq
                    if new_m[lq] == s2:
                        lq_at_s2 = lq

                if lq_at_s1 >= 0:
                    new_m[lq_at_s1] = s2
                if lq_at_s2 >= 0:
                    new_m[lq_at_s2] = s1

                new_g = g_cost + 1
                new_mk = mapping_key(new_m)
                child_key = (new_mk, new_exec_frozen)

                if child_key in visited and visited[child_key] <= new_g:
                    continue

                new_h = h_func(new_m, new_front_frozen)
                new_f = new_g + new_h

                if best_known_total < float('inf') and new_f > best_known_total * 1.2:
                    continue

                counter[0] += 1

                child_pair_swaps = defaultdict(float, parent_pair_swaps)
                for g in new_front_frozen:
                    gq1, gq2 = gates_2q[g]
                    pair_key = (min(gq1, gq2), max(gq1, gq2))
                    child_pair_swaps[pair_key] += 1.0

                swap_history[child_key] = child_pair_swaps
                heapq.heappush(open_heap, (new_f, counter[0], new_g, new_m, new_exec_frozen, new_front_frozen))

        if best_result is not None:
            return defaultdict(float, best_result[1]), best_result[0]
        else:
            return defaultdict(float), 0

    def simulate_routing_hybrid(m, rm, max_layers=20):
        astar_swaps, astar_total = astar_routing_early(m, rm)
        greedy_swaps, greedy_total, _ = simulate_routing_greedy(m, rm, max_layers=max_layers)

        combined_swaps = defaultdict(float)
        early_pairs = set()
        for g in early_layer_gates:
            gq1, gq2 = gates_2q[g]
            early_pairs.add((min(gq1, gq2), max(gq1, gq2)))

        for pair_key in early_pairs:
            if pair_key in astar_swaps:
                combined_swaps[pair_key] = astar_swaps[pair_key]

        for pair_key, count in greedy_swaps.items():
            if pair_key not in early_pairs:
                combined_swaps[pair_key] = count

        return combined_swaps, astar_total + greedy_total

    def simulate_routing_hybrid_bidir(m, rm, max_layers=20):
        astar_swaps, astar_total = astar_routing_early(m, rm)
        fwd_swaps, fwd_total, _ = simulate_routing_greedy(m, rm, max_layers=max_layers)
        bwd_swaps, bwd_total = simulate_routing_backward(m, rm, max_layers=max_layers)

        combined_swaps = defaultdict(float)
        early_pairs = set()
        for g in early_layer_gates:
            gq1, gq2 = gates_2q[g]
            early_pairs.add((min(gq1, gq2), max(gq1, gq2)))

        for pair_key in early_pairs:
            if pair_key in astar_swaps:
                combined_swaps[pair_key] = astar_swaps[pair_key]

        all_late_keys = (set(fwd_swaps.keys()) | set(bwd_swaps.keys())) - early_pairs
        for pair_key in all_late_keys:
            combined_swaps[pair_key] = 0.6 * fwd_swaps.get(pair_key, 0.0) + 0.4 * bwd_swaps.get(pair_key, 0.0)

        return combined_swaps, astar_total + (fwd_total + bwd_total) // 2

    # ---------------------------------------------------------------
    # Step 9: Seed generation
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
    # MIS seed
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
                    if len(candidates_list) > 1 and lq in logical_degree_rank:
                        lq_rn = logical_degree_rank[lq] / max_logical_rank
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
    # Spectral seed
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
    # Greedy placement
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

    def fill_unmapped(m, rm):
        unmapped = [q for q in range(num_q) if m[q] == -1]
        free = [pq for pq in range(num_q) if rm[pq] == -1]
        for lq, pq in zip(unmapped, free):
            m[lq] = pq
            rm[pq] = lq

    # ---------------------------------------------------------------
    # Local search
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
    # Perturbation modes with bandit selection
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
    # Step 10: Build initial mapping candidates
    # ---------------------------------------------------------------
    candidates = []

    for s_lq in seed_lqs:
        for s_pq in seed_pqs:
            m, rm = run_greedy_placement(s_lq, s_pq, static_nbrs, static_deg,
                                         weights=static_weight)
            fill_unmapped(m, rm)
            cost = compute_cost(m, static_weight)
            candidates.append((cost, m, rm))

    for s_pq in seed_pqs:
        cg_m, cg_rm = conflict_graph_seed(s_pq, static_nbrs, static_deg, static_weight)
        if cg_m is not None:
            fill_unmapped(cg_m, cg_rm)
            cost = compute_cost(cg_m, static_weight)
            candidates.append((cost, cg_m, cg_rm))

    spec_m, spec_rm = spectral_seed()
    if spec_m is not None:
        fill_unmapped(spec_m, spec_rm)
        spec_cost = compute_cost(spec_m, static_weight)
        candidates.append((spec_cost, spec_m, spec_rm))

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

    _, best_m, best_rm = candidates[0]
    best_m, best_rm = list(best_m), list(best_rm)
    local_search(best_m, best_rm, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=6)
    best_cost = compute_cost(best_m, static_weight)

    for idx in range(1, min(5, len(candidates))):
        _, m_c, rm_c = candidates[idx]
        m_c, rm_c = list(m_c), list(rm_c)
        local_search(m_c, rm_c, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=4)
        c = compute_cost(m_c, static_weight)
        if c < best_cost:
            best_cost = c
            best_m = list(m_c)
            best_rm = list(rm_c)

    # ---------------------------------------------------------------
    # Step 11: RSDIWR + ILS with Commutativity-Aware DAG Relinearization
    # ---------------------------------------------------------------
    time_budget = 25.0
    t_start = _time()
    n_rsdiwr = 4

    cur_m = list(best_m)
    cur_rm = list(best_rm)
    swap_counts = defaultdict(float)
    accumulated_routing_pressure = defaultdict(float)
    accumulated_routing_pressure_ref[0] = accumulated_routing_pressure

    for rsdiwr_iter in range(n_rsdiwr):
        rsdiwr_iter_ref[0] = rsdiwr_iter
        elapsed = _time() - t_start
        if elapsed > time_budget:
            break

        remaining_time = time_budget - elapsed
        remaining_iters = n_rsdiwr - rsdiwr_iter
        time_for_this_iter = remaining_time / max(remaining_iters, 1)
        n_ils = max(55, int(num_q * time_for_this_iter / 0.5)) if time_for_this_iter > 1.0 else max(30, num_q // 3)

        # Pressure-adaptive blending
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

        # Re-seed with routing-informed weights + routing-pressure MIS
        if rsdiwr_iter > 0 and swap_counts and (_time() - t_start) < time_budget * 0.7:
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
            if _time() - t_start > time_budget:
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

        # Routing simulation for RSDIWR feedback WITH bottleneck tracking
        if rsdiwr_iter < n_rsdiwr - 1:
            sim_depth = 14 + rsdiwr_iter * 7

            if rsdiwr_iter == 0 and early_layer_gates:
                swap_counts, _ = simulate_routing_hybrid(best_m, best_rm, max_layers=sim_depth)
                # Also run forward routing with bottleneck tracking for relinearization
                _, _, gate_bottlenecks = simulate_routing_greedy(best_m, best_rm, max_layers=sim_depth, track_bottlenecks=True)
            elif rsdiwr_iter == 1 and early_layer_gates:
                swap_counts, _ = simulate_routing_hybrid_bidir(best_m, best_rm, max_layers=sim_depth)
                _, _, gate_bottlenecks = simulate_routing_greedy(best_m, best_rm, max_layers=sim_depth, track_bottlenecks=True)
            else:
                swap_counts_bidir, _, gate_bottlenecks = simulate_routing_asymmetric_bidir(best_m, best_rm, max_layers=sim_depth, track_bottlenecks=True)
                swap_counts = swap_counts_bidir

            # Accumulate routing pressure
            for pair_key, cnt in swap_counts.items():
                accumulated_routing_pressure[pair_key] += cnt

            # === KEY NOVELTY: Commutativity-Aware DAG Relinearization ===
            # After routing simulation, reorder commuting adjacent gates
            # to reduce bottleneck SWAP counts
            if gate_bottlenecks and (_time() - t_start) < time_budget * 0.85:
                max_relinearize_swaps = min(20, max(5, len(gates_2q) // 10))
                dag_changed = relinearize_dag(gate_bottlenecks, best_m, best_rm,
                                              max_swaps=max_relinearize_swaps)

                if dag_changed:
                    # Recompute interaction weights with reordered DAG
                    # The gate_layer and critical_path are still valid since
                    # we only swapped commuting gates; but dep_count changed
                    # Update effective weights using new dep structure
                    pass  # dep_count already updated by recompute_topo_and_deps()

        cur_m = list(best_m)
        cur_rm = list(best_rm)
        cur_cost_nl = compute_cost_nonlinear(cur_m, eff_weights, alpha_end)

    # ---------------------------------------------------------------
    # Final: Set mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)