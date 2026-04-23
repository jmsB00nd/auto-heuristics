def init_mapping(self):
    import heapq

    num_q = self.num_qubits

    # --- Step 1: MAO on weighted logical interaction graph ---
    qig = self.qubit_interaction_graph
    logical_nodes = set(qig.keys())

    logical_mao = []
    if logical_nodes:
        # Start from the logical qubit with the highest total interaction weight
        start = max(logical_nodes, key=lambda q: sum(qig[q].values()))
        ordered = set()
        # key[v] = total weight of edges from v to the ordered set
        key = {v: 0 for v in logical_nodes}
        key[start] = float('inf')

        for _ in range(len(logical_nodes)):
            # Pick unordered vertex with max key
            best = max((v for v in logical_nodes if v not in ordered), key=lambda v: key[v])
            logical_mao.append(best)
            ordered.add(best)
            # Update keys of neighbors
            for neighbor, weight in qig[best].items():
                if neighbor not in ordered:
                    key[neighbor] += weight

    # --- Step 2: MAO on unweighted physical coupling graph ---
    backend = self.backend
    physical_nodes = set(backend.keys())

    physical_mao = []
    if physical_nodes:
        # Start from the physical qubit with highest degree
        start_p = max(physical_nodes, key=lambda q: len(backend[q]))
        ordered_p = set()
        key_p = {v: 0 for v in physical_nodes}
        key_p[start_p] = float('inf')

        for _ in range(len(physical_nodes)):
            best_p = max((v for v in physical_nodes if v not in ordered_p), key=lambda v: key_p[v])
            physical_mao.append(best_p)
            ordered_p.add(best_p)
            for neighbor in backend[best_p]:
                if neighbor not in ordered_p:
                    key_p[neighbor] += 1

    # --- Step 3: Align orderings ---
    self.mapping_dict = list(range(num_q))
    self.reverse_mapping_dict = list(range(num_q))

    used_physical = set()
    mapped_logical = set()

    # Map k-th logical MAO entry to k-th physical MAO entry
    k = min(len(logical_mao), len(physical_mao))
    for i in range(k):
        lq = logical_mao[i]
        pq = physical_mao[i]
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq
        used_physical.add(pq)
        mapped_logical.add(lq)

    # --- Step 4: Assign remaining logical qubits to remaining physical qubits ---
    remaining_physical = [p for p in range(num_q) if p not in used_physical]
    remaining_logical = [l for l in range(num_q) if l not in mapped_logical]

    for lq, pq in zip(remaining_logical, remaining_physical):
        self.mapping_dict[lq] = pq
        self.reverse_mapping_dict[pq] = lq

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)