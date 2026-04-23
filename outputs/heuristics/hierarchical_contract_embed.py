def init_mapping(self):
    import networkx as nx
    from collections import defaultdict

    num_q = self.num_qubits

    # --- Build logical interaction graph ---
    log_G = nx.Graph()
    log_G.add_nodes_from(range(num_q))
    for q1, neighbors in self.qubit_interaction_graph.items():
        for q2, w in neighbors.items():
            if q1 < q2:
                log_G.add_edge(q1, q2, weight=w)

    # --- Build physical coupling graph ---
    phys_G = nx.Graph()
    phys_G.add_nodes_from(range(num_q))
    for u, v in self.backend_connections:
        if not phys_G.has_edge(u, v):
            phys_G.add_edge(u, v, weight=1)

    # --- Coarsening helper ---
    def coarsen_graph(G):
        """Iteratively contract the heaviest edge. Returns list of (level_graph, merged_pair)."""
        hierarchy = []
        current = G.copy()
        # node_members[node] = set of original nodes it represents
        node_members = {n: {n} for n in current.nodes()}

        while current.number_of_nodes() > 1 and current.number_of_edges() > 0:
            # Find heaviest edge
            best_edge = max(current.edges(data=True), key=lambda e: e[2].get('weight', 1))
            u, v = best_edge[0], best_edge[1]

            # Record contraction
            hierarchy.append((dict(node_members), u, v))

            # Merge v into u
            merged_members = node_members[u] | node_members[v]

            # Combine edges
            for neighbor in list(current.neighbors(v)):
                if neighbor == u:
                    continue
                w_vn = current[v][neighbor].get('weight', 1)
                if current.has_edge(u, neighbor):
                    current[u][neighbor]['weight'] = current[u][neighbor].get('weight', 1) + w_vn
                else:
                    current.add_edge(u, neighbor, weight=w_vn)

            current.remove_node(v)
            node_members.pop(v, None)
            node_members[u] = merged_members

        # Final state
        hierarchy.append((dict(node_members), None, None))
        return hierarchy

    log_hierarchy = coarsen_graph(log_G)
    phys_hierarchy = coarsen_graph(phys_G)

    # --- Align hierarchy depths ---
    # We unwind from coarsest to finest. Pad the shorter hierarchy so both
    # have equal depth by repeating the coarsest level.
    max_depth = max(len(log_hierarchy), len(phys_hierarchy))
    while len(log_hierarchy) < max_depth:
        log_hierarchy.append(log_hierarchy[-1])
    while len(phys_hierarchy) < max_depth:
        phys_hierarchy.append(phys_hierarchy[-1])

    # --- At the coarsest level, assign logical super-nodes to physical super-nodes ---
    coarsest_log = log_hierarchy[-1][0]  # {super_node: {original nodes}}
    coarsest_phys = phys_hierarchy[-1][0]

    log_supers = sorted(coarsest_log.keys(), key=lambda k: -len(coarsest_log[k]))
    phys_supers = sorted(coarsest_phys.keys(), key=lambda k: -len(coarsest_phys[k]))

    # Greedy assignment: largest logical cluster -> largest physical cluster
    assignment = {}  # log_super -> phys_super
    used_phys = set()
    for ls in log_supers:
        best_ps = None
        for ps in phys_supers:
            if ps not in used_phys:
                if best_ps is None or len(coarsest_phys[ps]) > len(coarsest_phys.get(best_ps, set())):
                    best_ps = ps
        if best_ps is not None:
            assignment[ls] = best_ps
            used_phys.add(best_ps)

    # cluster_map: log_super -> set of physical originals assigned
    cluster_map = {}
    for ls, ps in assignment.items():
        cluster_map[frozenset(coarsest_log[ls])] = set(coarsest_phys[ps])

    # --- Unwind from coarsest to finest ---
    # Walk hierarchy backwards (from depth-2 downward to 0)
    for level in range(max_depth - 2, -1, -1):
        log_members_at_level, log_u, log_v = log_hierarchy[level]
        phys_members_at_level, phys_u, phys_v = phys_hierarchy[level]

        new_cluster_map = {}

        # For each current logical cluster, try to split it according to this level's contraction
        for log_set_frozen, phys_set in cluster_map.items():
            log_set = set(log_set_frozen)

            # Check if this level's logical contraction splits this cluster
            if log_u is not None and log_v is not None:
                members_u = log_members_at_level.get(log_u, set())
                members_v = log_members_at_level.get(log_v, set())

                part_a = log_set & members_u
                part_b = log_set & members_v
                remainder = log_set - part_a - part_b

                if part_a and part_b and len(phys_set) >= 2:
                    # Split physical set into two halves based on physical contraction
                    if phys_u is not None and phys_v is not None:
                        phys_members_u = phys_members_at_level.get(phys_u, set())
                        phys_members_v = phys_members_at_level.get(phys_v, set())
                        phys_a = phys_set & phys_members_u
                        phys_b = phys_set & phys_members_v
                        phys_rem = phys_set - phys_a - phys_b
                        # distribute physical remainder
                        for p in phys_rem:
                            if len(phys_a) <= len(phys_b):
                                phys_a.add(p)
                            else:
                                phys_b.add(p)
                    else:
                        phys_list = sorted(phys_set)
                        mid = len(phys_list) // 2
                        phys_a = set(phys_list[:mid])
                        phys_b = set(phys_list[mid:])

                    # Ensure physical partitions are non-empty
                    if not phys_a and phys_b:
                        p = next(iter(phys_b))
                        phys_a.add(p)
                        phys_b.discard(p)
                    elif not phys_b and phys_a:
                        p = next(iter(phys_a))
                        phys_b.add(p)
                        phys_a.discard(p)

                    # Match larger logical partition to larger physical partition
                    if len(part_a) > len(part_b):
                        if len(phys_a) < len(phys_b):
                            phys_a, phys_b = phys_b, phys_a
                    else:
                        if len(phys_b) < len(phys_a):
                            phys_a, phys_b = phys_b, phys_a

                    # Distribute remainder logical qubits
                    for q in remainder:
                        if len(part_a) <= len(part_b):
                            part_a.add(q)
                        else:
                            part_b.add(q)

                    new_cluster_map[frozenset(part_a)] = phys_a
                    new_cluster_map[frozenset(part_b)] = phys_b
                    continue

            # No split at this level — carry forward
            new_cluster_map[log_set_frozen] = phys_set

        cluster_map = new_cluster_map

    # --- Build final mapping from clusters ---
    mapping = list(range(num_q))  # identity fallback
    reverse_mapping = list(range(num_q))
    used_physical = set()
    mapped_logical = set()

    # For each cluster, assign logical -> physical greedily by interaction weight
    for log_set_frozen, phys_set in cluster_map.items():
        log_list = sorted(log_set_frozen,
                          key=lambda q: -self.logical_activity.get(q, 0))
        phys_list = sorted(phys_set,
                           key=lambda p: -self.physical_centrality.get(p, 0))

        for i, lq in enumerate(log_list):
            if i < len(phys_list):
                pq = phys_list[i]
                mapping[lq] = pq
                reverse_mapping[pq] = lq
                used_physical.add(pq)
                mapped_logical.add(lq)

    # Fallback: assign any unmapped logical qubits to remaining physical qubits
    free_physical = [p for p in range(num_q) if p not in used_physical]
    unmapped_logical = [q for q in range(num_q) if q not in mapped_logical]
    for lq, pq in zip(unmapped_logical, free_physical):
        mapping[lq] = pq
        reverse_mapping[pq] = lq

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)