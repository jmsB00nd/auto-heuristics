def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    N = self.num_qubits

    # --- 1. Collect logical 2-qubit interactions from self.access ---
    interactions = []
    logical_nodes = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            interactions.append((a, b))
            logical_nodes.add(a)
            logical_nodes.add(b)

    # --- 2. Build logical interaction graph (undirected, weighted) ---
    LG = nx.Graph()
    LG.add_nodes_from(logical_nodes)
    edge_w = defaultdict(int)
    for a, b in interactions:
        key = (a, b) if a < b else (b, a)
        edge_w[key] += 1
    for (a, b), w in edge_w.items():
        LG.add_edge(a, b, weight=w)

    # --- 3. Build physical coupling graph ---
    PG = nx.Graph()
    PG.add_nodes_from(range(N))
    for u, neighbors in self.backend.items():
        for v in neighbors:
            if u != v:
                PG.add_edge(u, v)

    # --- 4. k-core decomposition (drop self-loops if any to satisfy nx) ---
    def safe_core_number(G):
        H = G.copy()
        H.remove_edges_from(nx.selfloop_edges(H))
        try:
            return nx.core_number(H)
        except Exception:
            return {n: 0 for n in H.nodes()}

    logical_core = safe_core_number(LG)
    physical_core = safe_core_number(PG)

    # --- 5. Group nodes into shells by core number (descending) ---
    def group_by_core(core_dict):
        shells = defaultdict(list)
        for node, k in core_dict.items():
            shells[k].append(node)
        return shells

    logical_shells = group_by_core(logical_core)
    physical_shells = group_by_core(physical_core)

    logical_k_desc = sorted(logical_shells.keys(), reverse=True)
    physical_k_desc = sorted(physical_shells.keys(), reverse=True)

    # --- 6. Lockstep shell-by-shell greedy matching ---
    activity = self.logical_activity if self.logical_activity is not None else {}
    centrality = self.physical_centrality if self.physical_centrality is not None else {}

    mapping = [-1] * N
    reverse = [-1] * N
    used_physical = set()
    mapped_logical = set()

    # Maintain a pointer into physical shells; consume from deepest available downward.
    phys_pool = []  # ordered list of physical qubits, deepest core first
    for k in physical_k_desc:
        # within a shell, sort by centrality desc
        shell_sorted = sorted(
            physical_shells[k],
            key=lambda p: centrality.get(p, 0.0),
            reverse=True,
        )
        phys_pool.extend(shell_sorted)

    phys_idx = 0
    for k in logical_k_desc:
        log_shell = sorted(
            logical_shells[k],
            key=lambda q: activity.get(q, 0),
            reverse=True,
        )
        for lq in log_shell:
            # advance phys_idx past any already-used physicals
            while phys_idx < len(phys_pool) and phys_pool[phys_idx] in used_physical:
                phys_idx += 1
            if phys_idx >= len(phys_pool):
                break
            pq = phys_pool[phys_idx]
            mapping[lq] = pq
            reverse[pq] = lq
            used_physical.add(pq)
            mapped_logical.add(lq)
            phys_idx += 1

    # --- 7. Back-fill: any logical in self.access not yet mapped ---
    remaining_phys_sorted = sorted(
        (p for p in range(N) if p not in used_physical),
        key=lambda p: centrality.get(p, 0.0),
        reverse=True,
    )
    rp_iter = iter(remaining_phys_sorted)

    unmapped_logicals = sorted(
        (lq for lq in logical_nodes if lq not in mapped_logical),
        key=lambda q: activity.get(q, 0),
        reverse=True,
    )
    for lq in unmapped_logicals:
        try:
            pq = next(rp_iter)
        except StopIteration:
            break
        mapping[lq] = pq
        reverse[pq] = lq
        used_physical.add(pq)
        mapped_logical.add(lq)

    # --- 8. Identity-style fill for any remaining slots (idle logicals) ---
    remaining_phys = [p for p in range(N) if p not in used_physical]
    rp_set = set(remaining_phys)
    rp_list = list(remaining_phys)
    rp_pos = 0
    for lq in range(N):
        if mapping[lq] == -1:
            # prefer identity if available
            if lq in rp_set:
                mapping[lq] = lq
                reverse[lq] = lq
                rp_set.discard(lq)
            else:
                while rp_pos < len(rp_list) and rp_list[rp_pos] not in rp_set:
                    rp_pos += 1
                if rp_pos < len(rp_list):
                    pq = rp_list[rp_pos]
                    mapping[lq] = pq
                    reverse[pq] = lq
                    rp_set.discard(pq)
                    rp_pos += 1

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)