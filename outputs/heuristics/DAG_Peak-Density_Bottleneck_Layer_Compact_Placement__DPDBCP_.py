def init_mapping(self):
    """
    DAG Peak-Density Bottleneck Layer Compact Placement (DPDBCP)

    Identifies the circuit's bottleneck DAG layer — the layer with the maximum
    number of simultaneous 2Q gate interactions — then places the active qubits
    of that layer onto the densest connected subgraph of the hardware, ensuring
    maximum hardware connectivity at the circuit's most demanding moment.
    Remaining qubits are placed via interaction-weighted BFS expansion.
    """
    from collections import defaultdict, deque

    # -----------------------------------------------------------------------
    # Step 1: Compute DAG layers using the same dependency logic as DAG class
    # -----------------------------------------------------------------------
    schedule = sorted(self.access.keys())

    latest_writer = {}
    active_readers = {}
    read_since_writer = {}
    predecessors = defaultdict(set)

    for node in schedule:
        write_qubits = self.write_dict.get(node, [])
        all_qubits   = self.access[node]
        read_qubits  = [q for q in all_qubits if q not in write_qubits]

        for q in read_qubits:
            if q in latest_writer:
                predecessors[node].add(latest_writer[q])
            if q in active_readers:
                for old_reader in active_readers[q]:
                    if old_reader != node:
                        predecessors[node].add(old_reader)
                active_readers[q].clear()
            active_readers.setdefault(q, set()).add(node)
            read_since_writer[q] = True

        for q in write_qubits:
            if q in latest_writer and not read_since_writer.get(q, False):
                predecessors[node].add(latest_writer[q])
            if q in active_readers:
                for old_reader in active_readers[q]:
                    if old_reader != node:
                        predecessors[node].add(old_reader)
                active_readers[q].clear()
            latest_writer[q] = node
            read_since_writer[q] = False

    # Kahn's BFS: assign a topological level to each gate
    in_degree = {g: len(predecessors[g]) for g in schedule}
    successors_map = defaultdict(set)
    for g, preds in predecessors.items():
        for p in preds:
            successors_map[p].add(g)

    gate_level = {}
    queue = deque(g for g in schedule if in_degree[g] == 0)
    for g in queue:
        gate_level[g] = 0

    while queue:
        node = queue.popleft()
        for s in successors_map[node]:
            in_degree[s] -= 1
            gate_level[s] = max(gate_level.get(s, 0), gate_level[node] + 1)
            if in_degree[s] == 0:
                queue.append(s)

    layers = defaultdict(list)
    for gate, level in gate_level.items():
        layers[level].append(gate)

    # -----------------------------------------------------------------------
    # Step 2: Identify the bottleneck layer — max simultaneous 2Q gate count
    # -----------------------------------------------------------------------
    logical_qubit_set = set(q for qubits in self.access.values() for q in qubits)
    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    bottleneck_level = max(
        layers.keys(),
        key=lambda lvl: sum(1 for g in layers[lvl] if len(self.access[g]) == 2)
    )
    bottleneck_2q_gates = [g for g in layers[bottleneck_level] if len(self.access[g]) == 2]

    bottleneck_qubit_set = set(q for g in bottleneck_2q_gates for q in self.access[g])
    bottleneck_qubits = list(bottleneck_qubit_set)
    k = len(bottleneck_qubits)

    # -----------------------------------------------------------------------
    # Step 3: Find the densest connected hardware subgraph of size k
    #         Greedy expansion from every physical qubit as seed.
    # -----------------------------------------------------------------------
    def greedy_dense_subgraph(seed, size):
        subgraph = {seed}
        frontier = set(self.backend[seed])
        while len(subgraph) < size and frontier:
            best_p = max(frontier,
                         key=lambda p: sum(1 for nb in self.backend[p] if nb in subgraph))
            subgraph.add(best_p)
            frontier.discard(best_p)
            for nb in self.backend[best_p]:
                if nb not in subgraph:
                    frontier.add(nb)
        return subgraph

    def internal_edges(subgraph):
        return sum(1 for p in subgraph
                   for nb in self.backend[p] if nb in subgraph) // 2

    best_subgraph = None
    best_score = -1
    for seed in physical_qubits:
        sg = greedy_dense_subgraph(seed, k)
        if len(sg) < k:
            continue
        score = internal_edges(sg)
        if score > best_score:
            best_score = score
            best_subgraph = sg

    if best_subgraph is None or len(best_subgraph) < k:
        best_subgraph = set(physical_qubits[:k])

    dense_phys = list(best_subgraph)

    # -----------------------------------------------------------------------
    # Step 4: Map bottleneck qubits onto the dense subgraph
    #         Highest-interaction-degree logical <-> highest-internal-degree physical
    # -----------------------------------------------------------------------
    bn_degree = defaultdict(float)
    for g in bottleneck_2q_gates:
        for q in self.access[g]:
            bn_degree[q] += 1

    dense_set = set(dense_phys)
    phys_internal_deg = {
        p: sum(1 for nb in self.backend[p] if nb in dense_set)
        for p in dense_phys
    }

    lq_sorted   = sorted(bottleneck_qubits, key=lambda q: bn_degree[q],         reverse=True)
    phys_sorted = sorted(dense_phys,        key=lambda p: phys_internal_deg[p], reverse=True)

    lq_to_phys  = {}
    placed_phys = set()
    for lq, phys in zip(lq_sorted, phys_sorted):
        lq_to_phys[lq] = phys
        placed_phys.add(phys)

    # -----------------------------------------------------------------------
    # Step 5: Place remaining logical qubits via interaction-weighted BFS
    # -----------------------------------------------------------------------
    interaction_neighbors = defaultdict(dict)
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0) + 1
            interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0) + 1

    unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]

    while unplaced:
        next_lq = max(
            unplaced,
            key=lambda lq: sum(interaction_neighbors[lq].get(pl, 0) for pl in lq_to_phys)
        )

        candidates = [
            nb for phys in placed_phys
            for nb in self.backend[phys]
            if nb not in placed_phys
        ]
        if not candidates:
            candidates = [p for p in physical_qubits if p not in placed_phys]
        if not candidates:
            break

        def placement_cost(phys_c, _lq=next_lq):
            total = 0.0
            for pl_lq, pl_phys in lq_to_phys.items():
                w = interaction_neighbors[_lq].get(pl_lq, 0)
                if w > 0:
                    d = self.distance_matrix[phys_c][pl_phys]
                    total += w * (d if d != float('inf') else 1e9)
            return total

        best_phys = min(candidates, key=placement_cost)
        lq_to_phys[next_lq] = best_phys
        placed_phys.add(best_phys)
        unplaced.remove(next_lq)

    # -----------------------------------------------------------------------
    # Step 6: Build a strict 1-to-1 bijection over all num_qubits indices
    # -----------------------------------------------------------------------
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