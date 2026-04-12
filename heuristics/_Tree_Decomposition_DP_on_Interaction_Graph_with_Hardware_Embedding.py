def init_mapping(self):
    import math
    import random
    from collections import defaultdict, deque
    from time import time as _time

    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    dist = self.distance_matrix

    # ---------------------------------------------------------------
    # Step 1: Build DAG, critical path, interaction graph
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

    gate_layer = {g: 0 for g in all_gates}
    temp_in = {g: len(predecessors_dag[g]) for g in all_gates}
    queue = deque(g for g in all_gates if temp_in[g] == 0)
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
            critical_path[g] = max(critical_path[g], critical_path[s] + 1)

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

    dep_count = defaultdict(int)
    for g in reversed(topo_2q):
        for s in dag2q_succ[g]:
            dep_count[g] = max(dep_count[g], dep_count[s] + 1)

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

    interaction_adj = defaultdict(set)
    for (q1, q2) in static_weight:
        interaction_adj[q1].add(q2)
        interaction_adj[q2].add(q1)

    # ---------------------------------------------------------------
    # Step 2: Hardware graph properties
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
    # Step 3: Helper functions
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

    static_nbrs, static_deg = build_neighbors(static_weight)
    max_iw = max(static_weight.values()) if static_weight else 1.0

    def compute_cost(m, weights):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * dist[m[q1]][m[q2]]
        return cost

    def compute_cost_nonlinear(m, weights, alpha_exp):
        cost = 0.0
        for (q1, q2), w in weights.items():
            if m[q1] >= 0 and m[q2] >= 0:
                cost += w * (dist[m[q1]][m[q2]] ** alpha_exp)
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
                delta += w_a * ((dist[pq_b][pq_q] ** alpha_exp) - (dist[pq_a][pq_q] ** alpha_exp))
            w_b = nbrs.get(lq_b, {}).get(q, 0.0)
            if w_b > 0:
                delta += w_b * ((dist[pq_a][pq_q] ** alpha_exp) - (dist[pq_b][pq_q] ** alpha_exp))
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
    # Step 4: Tree Decomposition via Greedy Min-Degree Elimination
    # ---------------------------------------------------------------
    def compute_tree_decomposition(vertices, adj):
        remaining = set(vertices)
        work_adj = defaultdict(set)
        for v in vertices:
            for u in adj.get(v, set()):
                if u in remaining:
                    work_adj[v].add(u)
                    work_adj[u].add(v)

        bags = []
        elim_map = {}  # vertex -> bag index

        while remaining:
            min_v = min(remaining, key=lambda v: len(work_adj[v] & remaining))
            neighbors = work_adj[min_v] & remaining
            bag = frozenset({min_v} | neighbors)
            bag_idx = len(bags)
            bags.append(bag)
            elim_map[min_v] = bag_idx

            nbr_list = list(neighbors)
            for i in range(len(nbr_list)):
                for j in range(i + 1, len(nbr_list)):
                    work_adj[nbr_list[i]].add(nbr_list[j])
                    work_adj[nbr_list[j]].add(nbr_list[i])

            remaining.discard(min_v)
            for u in list(work_adj[min_v]):
                work_adj[u].discard(min_v)

        # Remove redundant bags (subsets of others)
        unique_indices = []
        for i in range(len(bags)):
            redundant = False
            for j in range(len(bags)):
                if i != j and bags[i].issubset(bags[j]):
                    redundant = True
                    break
            if not redundant:
                unique_indices.append(i)

        if not unique_indices:
            unique_indices = [0] if bags else []

        new_bags = [bags[i] for i in unique_indices]
        treewidth = max((len(b) for b in new_bags), default=0) - 1

        # Build tree via maximum-overlap spanning tree (Prim's)
        n_bags = len(new_bags)
        tree_children = defaultdict(list)
        if n_bags <= 1:
            return new_bags, tree_children, treewidth, 0

        import heapq
        in_tree = {0}
        heap = []
        for j in range(1, n_bags):
            overlap = len(new_bags[0] & new_bags[j])
            heapq.heappush(heap, (-overlap, 0, j))

        parent_of = {}
        while heap and len(in_tree) < n_bags:
            neg_ov, u, v = heapq.heappop(heap)
            if v in in_tree:
                continue
            in_tree.add(v)
            tree_children[u].append(v)
            parent_of[v] = u
            for j in range(n_bags):
                if j not in in_tree:
                    overlap = len(new_bags[v] & new_bags[j])
                    heapq.heappush(heap, (-overlap, v, j))

        return new_bags, tree_children, treewidth, 0

    # ---------------------------------------------------------------
    # Step 5: Spectral Pre-Embedding for Candidate Restriction
    # ---------------------------------------------------------------
    def spectral_pre_embedding():
        if len(interacting_logical) < 2:
            return {v: 0.0 for v in interacting_logical}

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

        def power_iter(mat, n_iter=120):
            v = [random.gauss(0, 1) for _ in range(n)]
            for _ in range(n_iter):
                new_v = [sum(mat[i][j] * v[j] for j in range(n)) for i in range(n)]
                norm = math.sqrt(sum(x * x for x in new_v)) or 1e-12
                v = [x / norm for x in new_v]
            return v

        max_diag = max(L[i][i] for i in range(n)) + 1.0
        shifted = [[-L[i][j] if i != j else max_diag - L[i][j] for j in range(n)] for i in range(n)]

        v1 = power_iter(shifted, 100)
        dot_v1 = sum(x * x for x in v1) or 1e-12
        lam1 = sum(v1[i] * sum(shifted[i][j] * v1[j] for j in range(n)) for i in range(n)) / dot_v1
        shifted2 = [row[:] for row in shifted]
        for i in range(n):
            for j in range(n):
                shifted2[i][j] -= lam1 * v1[i] * v1[j] / dot_v1

        fiedler = power_iter(shifted2, 130)
        return {interacting_logical[i]: fiedler[i] for i in range(n)}

    def hardware_bfs_order():
        start_pq = min(physical_qubits, key=lambda pq: phys_centrality[pq])
        visited = set([start_pq])
        order = []
        bfs_q = deque([start_pq])
        while bfs_q:
            pq = bfs_q.popleft()
            order.append(pq)
            for nb in sorted(hw_adj[pq]):
                if nb not in visited:
                    visited.add(nb)
                    bfs_q.append(nb)
        for pq in physical_qubits:
            if pq not in visited:
                order.append(pq)
        return order

    # ---------------------------------------------------------------
    # Step 6: Tree Decomposition DP with Beam Search
    # ---------------------------------------------------------------
    def tree_decomp_dp_seed():
        if len(interacting_logical) < 2:
            return None, None

        bags, tree_children, treewidth, root = compute_tree_decomposition(
            interacting_logical, interaction_adj)

        if not bags:
            return None, None

        # Fall back if treewidth too large
        if treewidth > 15:
            return None, None

        # Spectral embedding for candidate restriction
        embedding = spectral_pre_embedding()
        hw_order = hardware_bfs_order()
        sorted_by_spec = sorted(interacting_logical, key=lambda q: embedding.get(q, 0))
        n_interact = len(sorted_by_spec)
        n_phys = len(hw_order)
        offset = max(0, (n_phys - n_interact) // 2)

        rough_mapping = {}
        for i, lq in enumerate(sorted_by_spec):
            rough_mapping[lq] = hw_order[min(offset + i, n_phys - 1)]

        RADIUS = 3
        MAX_CAND = 25

        def get_bag_candidates(bag):
            bag_phys = [rough_mapping[lq] for lq in bag if lq in rough_mapping]
            if not bag_phys:
                return list(physical_qubits[:MAX_CAND])
            center = min(physical_qubits, key=lambda pq: sum(dist[pq][p] for p in bag_phys))
            candidates = set()
            frontier = {center}
            candidates.add(center)
            for _ in range(RADIUS):
                nxt = set()
                for pq in frontier:
                    for nb in hw_adj[pq]:
                        if nb not in candidates:
                            candidates.add(nb)
                            nxt.add(nb)
                frontier = nxt
            # Ensure enough candidates
            while len(candidates) < max(len(bag) + 3, 15):
                nxt = set()
                for pq in frontier:
                    for nb in hw_adj[pq]:
                        if nb not in candidates:
                            candidates.add(nb)
                            nxt.add(nb)
                if not nxt:
                    break
                frontier = nxt
            return sorted(candidates)

        # Build bottom-up order
        n_bags = len(bags)
        parent_of_bag = {}
        bottom_up = []
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                bottom_up.append(node)
            else:
                stack.append((node, True))
                for ch in tree_children.get(node, []):
                    parent_of_bag[ch] = node
                    stack.append((ch, False))

        # DP with beam search
        BEAM_WIDTH = 3000
        MAX_ENUM_PER_BAG = 200000

        bag_dp = {}

        for bag_idx in bottom_up:
            bag = bags[bag_idx]
            bag_list = sorted(bag)
            candidates = get_bag_candidates(bag)

            if len(candidates) < len(bag_list):
                candidates = sorted(set(candidates) | set(physical_qubits))

            # Separator with parent
            if bag_idx in parent_of_bag:
                sep = sorted(bag & bags[parent_of_bag[bag_idx]])
            else:
                sep = []

            sep_set = set(sep)
            internal_qs = [q for q in bag_list if q not in sep_set]

            # Internal edges
            internal_edges = []
            for (eq1, eq2), w in static_weight.items():
                if eq1 in bag and eq2 in bag:
                    internal_edges.append((eq1, eq2, w))

            child_indices = tree_children.get(bag_idx, [])
            child_seps = {}
            for ci in child_indices:
                child_seps[ci] = sorted(bag & bags[ci])

            n_bag = len(bag_list)

            # Estimate enumeration size
            enum_size = 1
            feasible = True
            for k in range(n_bag):
                enum_size *= (len(candidates) - k)
                if enum_size > MAX_ENUM_PER_BAG:
                    feasible = False
                    break

            results = {}

            if feasible and n_bag <= 7:
                # Exact enumeration via recursive backtracking
                assignment = [None] * n_bag
                bag_to_idx = {lq: i for i, lq in enumerate(bag_list)}

                def backtrack(idx, used):
                    if idx == n_bag:
                        assign_dict = {bag_list[k]: assignment[k] for k in range(n_bag)}
                        cost = 0.0
                        for eq1, eq2, w in internal_edges:
                            cost += w * dist[assign_dict[eq1]][assign_dict[eq2]]
                        for ci in child_indices:
                            cs = child_seps[ci]
                            cs_assign = tuple(assign_dict[q] for q in cs)
                            cdp = bag_dp.get(ci, {})
                            if cs_assign in cdp:
                                cost += cdp[cs_assign][0]
                            else:
                                return
                        sep_key = tuple(assign_dict[q] for q in sep) if sep else ()
                        if sep_key not in results or cost < results[sep_key][0]:
                            results[sep_key] = (cost, dict(assign_dict))
                        return

                    for pq in candidates:
                        if pq not in used:
                            assignment[idx] = pq
                            used.add(pq)
                            backtrack(idx + 1, used)
                            used.discard(pq)

                backtrack(0, set())

            else:
                # Beam search: enumerate separator assignments, then optimize internal
                if sep:
                    # For each separator assignment from candidates
                    n_sep = len(sep)
                    sep_assignments = []

                    def gen_sep_assigns(idx, used, partial):
                        if idx == n_sep:
                            sep_assignments.append(list(partial))
                            return
                        if len(sep_assignments) > BEAM_WIDTH * 3:
                            return
                        for pq in candidates:
                            if pq not in used:
                                partial.append(pq)
                                used.add(pq)
                                gen_sep_assigns(idx + 1, used, partial)
                                partial.pop()
                                used.discard(pq)

                    gen_sep_assigns(0, set(), [])

                    # Score separator assignments using child DP consistency
                    scored_sep = []
                    for sa in sep_assignments:
                        sep_dict = {sep[k]: sa[k] for k in range(n_sep)}
                        child_cost = 0.0
                        valid = True
                        for ci in child_indices:
                            cs = child_seps[ci]
                            cs_assign = tuple(sep_dict.get(q, -1) for q in cs)
                            if -1 in cs_assign:
                                continue
                            cdp = bag_dp.get(ci, {})
                            if cs_assign in cdp:
                                child_cost += cdp[cs_assign][0]
                            else:
                                valid = False
                                break
                        if valid:
                            scored_sep.append((child_cost, sa))

                    scored_sep.sort(key=lambda x: x[0])
                    scored_sep = scored_sep[:BEAM_WIDTH]

                    for _, sa in scored_sep:
                        sep_dict = {sep[k]: sa[k] for k in range(len(sep))}
                        used = set(sa)
                        remaining_cands = [pq for pq in candidates if pq not in used]

                        # Greedy assignment of internal qubits
                        assign_dict = dict(sep_dict)
                        for lq in internal_qs:
                            placed_partners = []
                            for partner in interaction_adj.get(lq, set()):
                                if partner in assign_dict:
                                    w = static_weight.get((min(lq, partner), max(lq, partner)), 0)
                                    if w > 0:
                                        placed_partners.append((w, assign_dict[partner]))
                            best_pq, best_sc = None, float('inf')
                            for pq in remaining_cands:
                                if pq in used:
                                    continue
                                if placed_partners:
                                    sc = sum(w * dist[pq][pp] for w, pp in placed_partners)
                                else:
                                    sc = phys_centrality.get(pq, 0)
                                if sc < best_sc:
                                    best_sc = sc
                                    best_pq = pq
                            if best_pq is None:
                                for pq in physical_qubits:
                                    if pq not in used:
                                        best_pq = pq
                                        break
                            if best_pq is not None:
                                assign_dict[lq] = best_pq
                                used.add(best_pq)

                        if len(assign_dict) == n_bag:
                            cost = 0.0
                            for eq1, eq2, w in internal_edges:
                                cost += w * dist[assign_dict[eq1]][assign_dict[eq2]]
                            for ci in child_indices:
                                cs = child_seps[ci]
                                cs_assign = tuple(assign_dict.get(q, -1) for q in cs)
                                cdp = bag_dp.get(ci, {})
                                if cs_assign in cdp:
                                    cost += cdp[cs_assign][0]

                            sep_key = tuple(sa)
                            if sep_key not in results or cost < results[sep_key][0]:
                                results[sep_key] = (cost, dict(assign_dict))

                else:
                    # Root bag with no separator
                    # Sample and greedy
                    for _ in range(min(BEAM_WIDTH, 5000)):
                        random.shuffle(candidates)
                        used = set()
                        assign_dict = {}
                        # Place by degree order
                        sorted_bag = sorted(bag_list, key=lambda q: logical_degree.get(q, 0), reverse=True)
                        for lq in sorted_bag:
                            placed_partners = []
                            for partner in interaction_adj.get(lq, set()):
                                if partner in assign_dict:
                                    w = static_weight.get((min(lq, partner), max(lq, partner)), 0)
                                    if w > 0:
                                        placed_partners.append((w, assign_dict[partner]))
                            best_pq, best_sc = None, float('inf')
                            for pq in candidates:
                                if pq in used:
                                    continue
                                if placed_partners:
                                    sc = sum(w * dist[pq][pp] for w, pp in placed_partners)
                                else:
                                    sc = phys_centrality.get(pq, 0) + random.random() * 0.01
                                if sc < best_sc:
                                    best_sc = sc
                                    best_pq = pq
                            if best_pq is not None:
                                assign_dict[lq] = best_pq
                                used.add(best_pq)

                        if len(assign_dict) == n_bag:
                            cost = 0.0
                            for eq1, eq2, w in internal_edges:
                                cost += w * dist[assign_dict[eq1]][assign_dict[eq2]]
                            for ci in child_indices:
                                cs = child_seps[ci]
                                cs_assign = tuple(assign_dict.get(q, -1) for q in cs)
                                cdp = bag_dp.get(ci, {})
                                if cs_assign in cdp:
                                    cost += cdp[cs_assign][0]
                            sep_key = ()
                            if sep_key not in results or cost < results[sep_key][0]:
                                results[sep_key] = (cost, dict(assign_dict))

            # Prune
            if len(results) > BEAM_WIDTH:
                sorted_r = sorted(results.items(), key=lambda x: x[1][0])
                results = dict(sorted_r[:BEAM_WIDTH])

            bag_dp[bag_idx] = results

        # Extract best from root
        root_dp = bag_dp.get(root, {})
        if not root_dp:
            return None, None

        best_key = min(root_dp, key=lambda k: root_dp[k][0])
        _, best_assign = root_dp[best_key]

        m = [-1] * num_q
        rm = [-1] * num_q
        used_phys = set()

        for lq, pq in best_assign.items():
            if m[lq] == -1 and pq not in used_phys:
                m[lq] = pq
                rm[pq] = lq
                used_phys.add(pq)

        # Top-down extraction for children
        top_q = deque([root])
        while top_q:
            bi = top_q.popleft()
            for ci in tree_children.get(bi, []):
                cs = sorted(bags[bi] & bags[ci])
                cs_assign = tuple(m[q] for q in cs)
                cdp = bag_dp.get(ci, {})
                if cs_assign in cdp:
                    _, ca = cdp[cs_assign]
                    for lq, pq in ca.items():
                        if m[lq] == -1 and pq not in used_phys:
                            m[lq] = pq
                            rm[pq] = lq
                            used_phys.add(pq)
                top_q.append(ci)

        # Place remaining interacting qubits greedily
        for lq in interacting_logical:
            if m[lq] == -1:
                placed = [(static_weight.get((min(lq, p), max(lq, p)), 0), m[p])
                          for p in interaction_adj.get(lq, set()) if m[p] >= 0]
                best_pq, best_sc = None, float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    if placed:
                        sc = sum(w * dist[pq][pp] for w, pp in placed if w > 0)
                    else:
                        sc = phys_centrality[pq]
                    if sc < best_sc:
                        best_sc = sc
                        best_pq = pq
                if best_pq is not None:
                    m[lq] = best_pq
                    rm[best_pq] = lq
                    used_phys.add(best_pq)

        return m, rm

    # ---------------------------------------------------------------
    # Step 7: MIS + Greedy Fallback Seed
    # ---------------------------------------------------------------
    def mis_greedy_seed(start_pq):
        if len(interacting_logical) < 2:
            return None, None

        top_partners = {}
        for lq in interacting_logical:
            nbr_list = sorted(static_nbrs.get(lq, {}).items(), key=lambda x: x[1], reverse=True)
            top_partners[lq] = set(p for p, _ in nbr_list[:3])

        conflict_adj = defaultdict(set)
        il = interacting_logical
        for i in range(len(il)):
            for j in range(i + 1, len(il)):
                s1, s2 = top_partners.get(il[i], set()), top_partners.get(il[j], set())
                union = len(s1 | s2)
                if union > 0 and len(s1 & s2) / union > 0.5:
                    conflict_adj[il[i]].add(il[j])
                    conflict_adj[il[j]].add(il[i])

        sorted_by_deg = sorted(interacting_logical, key=lambda q: logical_degree[q], reverse=True)
        mis, excluded = set(), set()
        for lq in sorted_by_deg:
            if lq not in excluded:
                mis.add(lq)
                excluded.update(conflict_adj.get(lq, set()))

        non_mis = [lq for lq in sorted_by_deg if lq not in mis]
        mis_ordered = sorted(mis, key=lambda q: logical_degree[q], reverse=True)

        m = [-1] * num_q
        rm = [-1] * num_q
        used_phys = set()

        if mis_ordered:
            m[mis_ordered[0]] = start_pq
            rm[start_pq] = mis_ordered[0]
            used_phys.add(start_pq)

            for lq in mis_ordered[1:]:
                nbrs_placed = {plq: static_nbrs[lq][plq] for plq in mis if m[plq] >= 0 and plq in static_nbrs.get(lq, {})}
                if nbrs_placed:
                    best_pq, best_sc = None, float('inf')
                    for pq in physical_qubits:
                        if pq in used_phys:
                            continue
                        sc = sum(iw * dist[pq][m[plq]] * (0.85 if m[plq] in hw_adj[pq] else 1.0) for plq, iw in nbrs_placed.items())
                        if sc < best_sc:
                            best_sc = sc
                            best_pq = pq
                else:
                    best_pq = min((pq for pq in physical_qubits if pq not in used_phys), key=lambda pq: phys_centrality[pq], default=None)
                if best_pq is not None:
                    m[lq] = best_pq
                    rm[best_pq] = lq
                    used_phys.add(best_pq)

        placed_set = set(lq for lq in interacting_logical if m[lq] >= 0)
        for lq in non_mis:
            nbrs_placed = {plq: static_nbrs.get(lq, {}).get(plq, 0) for plq in placed_set if static_nbrs.get(lq, {}).get(plq, 0) > 0}
            if nbrs_placed:
                best_pq, best_sc = None, float('inf')
                for pq in physical_qubits:
                    if pq in used_phys:
                        continue
                    sc = sum(iw * dist[pq][m[plq]] * (0.85 if m[plq] in hw_adj[pq] else 1.0) for plq, iw in nbrs_placed.items())
                    if sc < best_sc:
                        best_sc = sc
                        best_pq = pq
            else:
                best_pq = min((pq for pq in physical_qubits if pq not in used_phys), key=lambda pq: phys_centrality[pq], default=None)
            if best_pq is not None:
                m[lq] = best_pq
                rm[best_pq] = lq
                used_phys.add(best_pq)
                placed_set.add(lq)

        return m, rm

    # ---------------------------------------------------------------
    # Step 8: Local Search
    # ---------------------------------------------------------------
    def local_search(m, rm, nbrs, weights, alpha_exp, max_rounds=5):
        if len(interacting_logical) <= 1:
            return compute_cost_nonlinear(m, weights, alpha_exp)
        for _ in range(max_rounds):
            best_d, best_pair = 0.0, None
            active_phys = [m[q] for q in interacting_logical if m[q] >= 0]
            for pq1 in active_phys:
                for pq2 in hw_adj.get(pq1, set()):
                    d = delta_swap_cost_nonlinear(m, rm, pq1, pq2, nbrs, alpha_exp)
                    if d < best_d:
                        best_d = d
                        best_pair = (pq1, pq2)
            n_rand = min(150, len(interacting_logical) * 3)
            for _ in range(n_rand):
                if len(interacting_logical) >= 2:
                    i, j = random.sample(range(len(interacting_logical)), 2)
                    pq1, pq2 = m[interacting_logical[i]], m[interacting_logical[j]]
                    if pq1 >= 0 and pq2 >= 0:
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
    # Step 9: Routing Simulation for RSDIWR
    # ---------------------------------------------------------------
    def simulate_routing(m, rm, max_layers=20):
        sim_m, sim_rm = list(m), list(rm)
        swap_counts = defaultdict(float)
        if not gates_2q:
            return swap_counts
        pred_rem = {g: len(dag2q_pred[g]) for g in gates_2q}
        front = set(g for g in gates_2q if pred_rem[g] == 0)
        layers_done = 0
        while front and layers_done < max_layers:
            executable = [g for g in front if (sim_m[gates_2q[g][0]], sim_m[gates_2q[g][1]]) in self.backend_connections or (sim_m[gates_2q[g][1]], sim_m[gates_2q[g][0]]) in self.backend_connections]
            if executable:
                for g in executable:
                    front.discard(g)
                    for s in dag2q_succ[g]:
                        pred_rem[s] -= 1
                        if pred_rem[s] == 0:
                            front.add(s)
                layers_done += 1
                continue
            active_phys = set()
            for g in front:
                active_phys.add(sim_m[gates_2q[g][0]])
                active_phys.add(sim_m[gates_2q[g][1]])
            cands = set()
            for pq in active_phys:
                for nb in self.backend.get(pq, []):
                    cands.add((min(pq, nb), max(pq, nb)))
            best_sw, best_sc = None, float('inf')
            for (s1, s2) in cands:
                l1, l2 = sim_rm[s1], sim_rm[s2]
                sc = 0.0
                for g in front:
                    gq1, gq2 = gates_2q[g]
                    p1, p2 = sim_m[gq1], sim_m[gq2]
                    if gq1 == l1: p1 = s2
                    elif gq1 == l2: p1 = s1
                    if gq2 == l1: p2 = s2
                    elif gq2 == l2: p2 = s1
                    sc += (dep_count.get(g, 0) + 1) * dist[p1][p2]
                if sc < best_sc:
                    best_sc = sc
                    best_sw = (s1, s2)
            if best_sw is None:
                break
            s1, s2 = best_sw
            l1, l2 = sim_rm[s1], sim_rm[s2]
            sim_m[l1], sim_m[l2] = s2, s1
            sim_rm[s1], sim_rm[s2] = l2, l1
            for g in front:
                pair = (min(gates_2q[g][0], gates_2q[g][1]), max(gates_2q[g][0], gates_2q[g][1]))
                swap_counts[pair] += 1.0
        return swap_counts

    # ---------------------------------------------------------------
    # Step 10: Build Seed Candidates
    # ---------------------------------------------------------------
    candidates = []

    # Tree Decomposition DP seed (the novel contribution)
    t_start = _time()
    td_m, td_rm = tree_decomp_dp_seed()
    if td_m is not None:
        fill_unmapped(td_m, td_rm)
        td_cost = compute_cost(td_m, static_weight)
        candidates.append((td_cost, td_m, td_rm))

    # MIS + Greedy fallback seeds
    phys_by_centrality = sorted(physical_qubits, key=lambda pq: phys_centrality[pq])
    seed_pqs = phys_by_centrality[:min(3, len(phys_by_centrality))]

    for sp in seed_pqs:
        mg_m, mg_rm = mis_greedy_seed(sp)
        if mg_m is not None:
            fill_unmapped(mg_m, mg_rm)
            c = compute_cost(mg_m, static_weight)
            candidates.append((c, mg_m, mg_rm))

    # Random seed for diversity
    m_rand = [-1] * num_q
    rm_rand = [-1] * num_q
    shuffled = list(physical_qubits)
    random.shuffle(shuffled)
    for i, lq in enumerate(logical_qubits):
        m_rand[lq] = shuffled[i]
        rm_rand[shuffled[i]] = lq
    fill_unmapped(m_rand, rm_rand)
    candidates.append((compute_cost(m_rand, static_weight), m_rand, rm_rand))

    if not candidates:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    candidates.sort(key=lambda x: x[0])

    # Refine top candidates with local search
    _, best_m, best_rm = candidates[0]
    best_m, best_rm = list(best_m), list(best_rm)
    local_search(best_m, best_rm, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=6)
    best_cost = compute_cost(best_m, static_weight)

    for idx in range(1, min(5, len(candidates))):
        _, mc, rmc = candidates[idx]
        mc, rmc = list(mc), list(rmc)
        local_search(mc, rmc, static_nbrs, static_weight, alpha_exp=2.0, max_rounds=4)
        c = compute_cost(mc, static_weight)
        if c < best_cost:
            best_cost = c
            best_m, best_rm = list(mc), list(rmc)

    # ---------------------------------------------------------------
    # Step 11: ILS + SA + RSDIWR Refinement
    # ---------------------------------------------------------------
    time_budget = 25.0
    n_rsdiwr = 4

    def perturb(m, rm, nbrs, weights, alpha_exp):
        mode = random.randint(0, 4)
        if mode == 0 and len(interacting_logical) >= 2:
            lqs = random.sample(interacting_logical, 2)
            do_swap(m, rm, m[lqs[0]], m[lqs[1]])
        elif mode == 1:
            k = min(random.randint(3, 5), len(interacting_logical))
            if k >= 2:
                lqs = random.sample(interacting_logical, k)
                pps = [m[lq] for lq in lqs]
                random.shuffle(pps)
                for lq in lqs:
                    rm[m[lq]] = -1
                for lq, pq in zip(lqs, pps):
                    m[lq] = pq
                    rm[pq] = lq
        elif mode == 2 and weights:
            pair_costs = [(w * (dist[m[q1]][m[q2]] ** alpha_exp), q1, q2) for (q1, q2), w in weights.items() if m[q1] >= 0 and m[q2] >= 0]
            if pair_costs:
                pair_costs.sort(reverse=True)
                _, tq1, tq2 = pair_costs[random.randint(0, min(2, len(pair_costs) - 1))]
                adj_list = list(hw_adj.get(m[tq2], set()))
                if adj_list:
                    tgt = random.choice(adj_list)
                    do_swap(m, rm, m[tq1], tgt)
        elif mode == 3 and len(interacting_logical) >= 3 and nbrs:
            k = min(random.randint(3, 7), len(interacting_logical))
            qcost = {}
            for lq in interacting_logical:
                c = sum(w * (dist[m[lq]][m[p]] ** alpha_exp) for p, w in nbrs.get(lq, {}).items() if m[p] >= 0 and m[lq] >= 0)
                qcost[lq] = c
            top = sorted(qcost, key=lambda q: qcost[q], reverse=True)[:max(k, len(interacting_logical) // 2)]
            subset = random.sample(top, min(k, len(top)))
            freed = []
            for lq in subset:
                freed.append(m[lq])
                rm[m[lq]] = -1
                m[lq] = -1
            placed = set(lq for lq in interacting_logical if m[lq] >= 0)
            for lq in sorted(subset, key=lambda q: qcost[q], reverse=True):
                best_pq, best_sc = None, float('inf')
                for pq in freed:
                    if rm[pq] != -1:
                        continue
                    sc = sum(w * (dist[pq][m[p]] ** alpha_exp) for p, w in nbrs.get(lq, {}).items() if p in placed and m[p] >= 0)
                    if sc < best_sc:
                        best_sc = sc
                        best_pq = pq
                if best_pq is None:
                    for pq in freed:
                        if rm[pq] == -1:
                            best_pq = pq
                            break
                if best_pq is not None:
                    m[lq] = best_pq
                    rm[best_pq] = lq
                    placed.add(lq)
        else:
            if len(interacting_logical) >= 2:
                lqs = random.sample(interacting_logical, 2)
                do_swap(m, rm, m[lqs[0]], m[lqs[1]])

    cur_m, cur_rm = list(best_m), list(best_rm)
    swap_counts = defaultdict(float)

    for rsdiwr_iter in range(n_rsdiwr):
        elapsed = _time() - t_start
        if elapsed > time_budget:
            break

        remaining_time = time_budget - elapsed
        remaining_iters = n_rsdiwr - rsdiwr_iter
        time_for_iter = remaining_time / max(remaining_iters, 1)
        n_ils = max(40, int(num_q * time_for_iter / 0.5)) if time_for_iter > 1.0 else max(25, num_q // 3)

        if rsdiwr_iter == 0 or not swap_counts:
            eff_weights = dict(static_weight)
        else:
            max_sw = max(swap_counts.values()) if swap_counts else 1.0
            scale = max(static_weight.values()) / max(max_sw, 1e-10)
            alpha_blend = max(0.3, 1.0 - 0.3 * rsdiwr_iter)
            eff_weights = defaultdict(float)
            for key in set(static_weight.keys()) | set(swap_counts.keys()):
                eff_weights[key] = alpha_blend * static_weight.get(key, 0) + (1 - alpha_blend) * swap_counts.get(key, 0) * scale

        eff_nbrs, _ = build_neighbors(eff_weights)

        alpha_start, alpha_end = 2.0, 1.0
        cur_cost_nl = local_search(cur_m, cur_rm, eff_nbrs, eff_weights, alpha_exp=alpha_start, max_rounds=5)

        sc = compute_cost(cur_m, static_weight)
        if sc < best_cost:
            best_cost = sc
            best_m, best_rm = list(cur_m), list(cur_rm)

        T = max(cur_cost_nl * 0.05, 1.0)
        T_init = T
        alpha_sa = 0.94
        reheat = max(n_ils // 4, 8)

        for ils_iter in range(n_ils):
            if _time() - t_start > time_budget:
                break

            progress = ils_iter / max(n_ils - 1, 1)
            alpha_exp = alpha_start + (alpha_end - alpha_start) * progress

            saved_m, saved_rm = list(cur_m), list(cur_rm)
            saved_cost = cur_cost_nl

            perturb(cur_m, cur_rm, eff_nbrs, eff_weights, alpha_exp)
            new_cost = local_search(cur_m, cur_rm, eff_nbrs, eff_weights, alpha_exp=alpha_exp, max_rounds=3)

            improvement = saved_cost - new_cost
            if improvement > 0:
                cur_cost_nl = new_cost
                sc2 = compute_cost(cur_m, static_weight)
                if sc2 < best_cost:
                    best_cost = sc2
                    best_m, best_rm = list(cur_m), list(cur_rm)
            elif random.random() < math.exp(min(0, improvement / max(T, 1e-10))):
                cur_cost_nl = new_cost
            else:
                cur_m[:] = saved_m
                cur_rm[:] = saved_rm
                cur_cost_nl = saved_cost

            T *= alpha_sa
            if (ils_iter + 1) % reheat == 0:
                T = max(T, T_init * 0.35)

        if rsdiwr_iter < n_rsdiwr - 1:
            sim_depth = 12 + rsdiwr_iter * 6
            swap_counts = simulate_routing(best_m, best_rm, max_layers=sim_depth)

        cur_m, cur_rm = list(best_m), list(best_rm)
        cur_cost_nl = compute_cost_nonlinear(cur_m, eff_weights, alpha_end)

    # ---------------------------------------------------------------
    # Step 12: Set final mapping
    # ---------------------------------------------------------------
    self.mapping_dict = best_m
    self.reverse_mapping_dict = best_rm

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)