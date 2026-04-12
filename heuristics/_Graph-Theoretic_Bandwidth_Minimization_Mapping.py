def init_mapping(self):
    """
    Graph-Theoretic Bandwidth Minimization Mapping.
    
    Computes Reverse Cuthill-McKee (RCM) orderings of both the logical
    interaction graph and the hardware coupling graph, then maps the k-th
    logical qubit in the logical RCM ordering to the k-th physical qubit
    in the hardware RCM ordering. This aligns the narrow-bandwidth
    structures of both graphs, reducing expected routing overhead.
    """
    from collections import defaultdict, deque

    # --- Step 1: Build logical interaction graph as adjacency list ---
    logical_adj = defaultdict(set)
    logical_qubit_set = set()

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            logical_adj[q1].add(q2)
            logical_adj[q2].add(q1)

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # Fallback: trivial identity if no logical qubits
    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # --- Step 2: Reverse Cuthill-McKee ordering ---
    def reverse_cuthill_mckee(adj, nodes):
        """
        Compute RCM ordering of the graph defined by adj over the given nodes.
        Handles disconnected components by processing each separately.
        """
        node_set = set(nodes)
        visited = set()
        cm_order = []

        # Degree within the subgraph
        def degree(v):
            return len(adj[v] & node_set)

        # Process each connected component, starting from the node with
        # minimum degree (peripheral node heuristic for good bandwidth).
        remaining = sorted(nodes, key=lambda v: degree(v))

        for start_candidate in remaining:
            if start_candidate in visited:
                continue
            # BFS from start_candidate, enqueuing neighbors in order of
            # increasing degree (standard Cuthill-McKee).
            queue = deque([start_candidate])
            visited.add(start_candidate)
            component_order = []

            while queue:
                node = queue.popleft()
                component_order.append(node)
                neighbors = sorted(
                    [n for n in adj[node] if n in node_set and n not in visited],
                    key=lambda v: degree(v)
                )
                for nb in neighbors:
                    visited.add(nb)
                    queue.append(nb)

            cm_order.extend(component_order)

        # Reverse the CM ordering to get RCM
        return list(reversed(cm_order))

    # --- Step 3: Compute RCM ordering of logical interaction graph ---
    logical_rcm = reverse_cuthill_mckee(logical_adj, logical_qubits)

    # --- Step 4: Compute RCM ordering of hardware coupling graph ---
    hardware_rcm = reverse_cuthill_mckee(self.backend, physical_qubits)

    # --- Step 5: Map k-th logical qubit in logical RCM to k-th physical
    #             qubit in hardware RCM ---
    lq_to_phys = {}
    for i, lq in enumerate(logical_rcm):
        if i < len(hardware_rcm):
            lq_to_phys[lq] = hardware_rcm[i]

    # --- Step 6: Build strict 1-to-1 bijection over all num_qubits indices ---
    mapping_dict = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]
        mapping_dict[lq] = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys] = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)