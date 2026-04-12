def init_mapping(self):
    from collections import defaultdict, deque
    from itertools import permutations, combinations

    # ── Step 1: Build canonical weighted interaction graph ──────────────────
    # F[(min_q, max_q)] = total number of 2-qubit gates between the pair
    F = defaultdict(float)
    logical_qubit_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            F[(min(q1, q2), max(q1, q2))] += 1

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Trivial fallback when the circuit has no gates
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Mutable adjacency used only during the coarsening phase
    adj = defaultdict(lambda: defaultdict(float))
    for (q1, q2), w in F.items():
        adj[q1][q2] += w
        adj[q2][q1] += w

    # node_members[id] = list of original logical qubits contained in this (super-)node
    next_id = max(logical_qubits) + 1
    node_members = {q: [q] for q in logical_qubits}
    coarsen_history = []          # ordered list of (super_id, child1_id, child2_id)
    active_nodes = set(logical_qubits)

    # ── Step 2: Iterative coarsening until |V| ≤ 4 ────────────────────────
    while len(active_nodes) > 4:
        # Greedily pick the maximum-weight edge among active nodes
        best_u = best_v = None
        best_w = -1
        for u in active_nodes:
            for v, w in adj[u].items():
                if v in active_nodes and u < v and w > best_w:
                    best_w, best_u, best_v = w, u, v

        # Fallback when no edges exist (isolated qubits): merge arbitrary pair
        if best_u is None:
            nl = sorted(active_nodes)
            best_u, best_v = nl[0], nl[1]

        s = next_id; next_id += 1
        node_members[s] = node_members[best_u] + node_members[best_v]
        coarsen_history.append((s, best_u, best_v))

        # Build super-node edges: sum contributions from both children
        new_edges = defaultdict(float)
        for nb, w in adj[best_u].items():
            if nb != best_v and nb in active_nodes:
                new_edges[nb] += w
        for nb, w in adj[best_v].items():
            if nb != best_u and nb in active_nodes:
                new_edges[nb] += w
        adj[s] = new_edges
        for nb in new_edges:
            adj[nb][s] = new_edges[nb]
            adj[nb].pop(best_u, None)
            adj[nb].pop(best_v, None)

        active_nodes.discard(best_u)
        active_nodes.discard(best_v)
        active_nodes.add(s)

    # ── Step 3: Bottom-level exhaustive placement (≤ 4 nodes → ≤ 4! = 24 perms) ─
    coarse_nodes = sorted(active_nodes)
    n_coarse = len(coarse_nodes)

    def mean_dist(p):
        vals = [self.distance_matrix[p][o] for o in physical_qubits
                if o != p and self.distance_matrix[p][o] != float('inf')]
        return sum(vals) / len(vals) if vals else float('inf')

    # Restrict exhaustive search to the most globally-central physical qubits
    k = min(max(n_coarse * 3, 10), len(physical_qubits))
    central_phys = sorted(physical_qubits, key=mean_dist)[:k]

    best_cost = float('inf')
    node_to_phys = {coarse_nodes[i]: central_phys[i] for i in range(n_coarse)}

    for phys_subset in combinations(central_phys, n_coarse):
        for perm in permutations(phys_subset):
            cost = sum(
                adj[coarse_nodes[i]].get(coarse_nodes[j], 0.0) *
                self.distance_matrix[perm[i]][perm[j]]
                for i in range(n_coarse) for j in range(i + 1, n_coarse)
            )
            if cost < best_cost:
                best_cost = cost
                node_to_phys = {coarse_nodes[i]: perm[i] for i in range(n_coarse)}

    assigned_phys = set(node_to_phys.values())

    # ── Step 4: Uncoarsening with hardware-aware BFS + local refinement ────

    def bfs_nearest_free(start, occupied):
        """BFS on the hardware graph to find the nearest unoccupied physical qubit."""
        visited = {start}
        q = deque([start])
        while q:
            curr = q.popleft()
            if curr not in occupied:
                return curr
            for nb in self.backend[curr]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)
        return None

    def grp_cost(grp_lqs, grp_phys, other_items):
        """
        F-weighted distance from a group of logical qubits (all sharing grp_phys)
        to every other placed group.  Used for comparative local refinement only.
        """
        cost = 0.0
        for lq in grp_lqs:
            for o_lqs, o_phys in other_items:
                d = self.distance_matrix[grp_phys][o_phys]
                for o_lq in o_lqs:
                    w = F.get((min(lq, o_lq), max(lq, o_lq)), 0.0)
                    if w > 0:
                        cost += w * d
        return cost

    def other_items_except(*exclude_ids):
        """Return [(members_list, phys), …] for all nodes not in exclude_ids."""
        return [
            (node_members.get(nid, [nid]), ph)
            for nid, ph in node_to_phys.items()
            if nid not in exclude_ids
        ]

    for (s, child1, child2) in reversed(coarsen_history):
        if s not in node_to_phys:
            continue

        phys_s = node_to_phys.pop(s)

        # child1 inherits the super-node's physical location
        node_to_phys[child1] = phys_s

        # child2: BFS for nearest free physical qubit from phys_s
        phys_child2 = bfs_nearest_free(phys_s, assigned_phys)
        if phys_child2 is None:
            remaining = [p for p in physical_qubits if p not in assigned_phys]
            phys_child2 = remaining[0] if remaining else phys_s
        node_to_phys[child2] = phys_child2
        assigned_phys.add(phys_child2)

        c1_lqs = node_members[child1]
        c2_lqs = node_members[child2]
        others = other_items_except(child1, child2)

        # Optionally swap child1/child2 if it reduces interaction cost
        # (child1↔child2 internal distance is symmetric → cancels; compare vs rest only)
        c_norm = grp_cost(c1_lqs, phys_s, others) + grp_cost(c2_lqs, phys_child2, others)
        c_swap = grp_cost(c1_lqs, phys_child2, others) + grp_cost(c2_lqs, phys_s, others)
        if c_swap < c_norm:
            node_to_phys[child1] = phys_child2
            node_to_phys[child2] = phys_s

        # Extended local refinement: try swapping each recently placed node
        # with its hardware neighbors if it reduces cost
        phys_to_node = {ph: nid for nid, ph in node_to_phys.items()}
        for focus in [child1, child2]:
            fp = node_to_phys[focus]
            f_lqs = node_members[focus]
            for hw_nb in self.backend[fp]:
                if hw_nb not in assigned_phys:
                    continue
                other_node = phys_to_node.get(hw_nb)
                if other_node is None or other_node == focus:
                    continue
                o_lqs = node_members[other_node]
                oth = other_items_except(focus, other_node)
                before = grp_cost(f_lqs, fp, oth) + grp_cost(o_lqs, hw_nb, oth)
                after  = grp_cost(f_lqs, hw_nb, oth) + grp_cost(o_lqs, fp, oth)
                if after < before:
                    node_to_phys[focus] = hw_nb
                    node_to_phys[other_node] = fp
                    phys_to_node[hw_nb] = focus
                    phys_to_node[fp] = other_node
                    fp = hw_nb
                    break

    # ── Step 5: Build strict 1-to-1 bijection ──────────────────────────────
    lq_to_phys = {}
    for node_id, phys in node_to_phys.items():
        for lq in node_members.get(node_id, [node_id]):
            if lq in logical_qubit_set and lq not in lq_to_phys:
                lq_to_phys[lq] = phys

    # Safety: place any leftover logical qubits (should not arise in practice)
    used_phys = set(lq_to_phys.values())
    free_phys = [p for p in physical_qubits if p not in used_phys]
    for lq in logical_qubits:
        if lq not in lq_to_phys and free_phys:
            lq_to_phys[lq] = free_phys.pop(0)

    # Populate mapping arrays via swap-in to preserve global bijectivity
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))
    for lq, target_phys in lq_to_phys.items():
        cur_phys = mapping_dict[lq]
        if cur_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = cur_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[cur_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)