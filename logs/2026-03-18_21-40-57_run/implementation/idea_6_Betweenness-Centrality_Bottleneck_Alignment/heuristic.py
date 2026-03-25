def init_mapping(self):
    from collections import defaultdict, deque

    # --- Step 1: Build Qubit Interaction Graph (QIG) ---
    logical_qubit_set = set()
    qig = defaultdict(set)

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            qig[q1].add(q2)
            qig[q2].add(q1)

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    if not logical_qubits:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Ensure all logical qubits appear in the QIG adjacency (even isolates)
    for q in logical_qubits:
        if q not in qig:
            qig[q] = set()

    # --- Step 2: Brandes' Betweenness Centrality (unweighted BFS) ---
    def brandes_bc(nodes, adj):
        """
        Compute unnormalized betweenness centrality for all nodes via
        Brandes' O(V*E) algorithm.
        """
        bc = {n: 0.0 for n in nodes}
        for s in nodes:
            stack = []
            pred  = {n: [] for n in nodes}
            sigma = {n: 0   for n in nodes}
            dist  = {n: -1  for n in nodes}
            sigma[s] = 1
            dist[s]  = 0
            queue = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)
                for w in adj[v]:
                    if dist[w] < 0:          # first visit
                        queue.append(w)
                        dist[w] = dist[v] + 1
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append(v)

            delta = {n: 0.0 for n in nodes}
            while stack:
                w = stack.pop()
                for v in pred[w]:
                    if sigma[w] > 0:
                        delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
                if w != s:
                    bc[w] += delta[w]
        return bc

    # --- Step 3: Compute BC for logical qubits on QIG ---
    logical_bc = brandes_bc(logical_qubits, qig)

    # --- Step 4: Compute BC for physical qubits on hardware graph ---
    physical_bc = brandes_bc(physical_qubits, self.backend)

    # --- Step 5: Rank-align — highest BC logical → highest BC physical ---
    sorted_logical  = sorted(logical_qubits,  key=lambda q: logical_bc[q],  reverse=True)
    sorted_physical = sorted(physical_qubits, key=lambda p: physical_bc[p], reverse=True)

    lq_to_phys = {lq: sorted_physical[i] for i, lq in enumerate(sorted_logical)}

    # --- Step 6: Build strict 1-to-1 bijection lists ---
    # Start from the identity permutation and apply each assignment as a
    # transposition, keeping the mapping consistent at every step.
    mapping_dict         = list(range(self.num_qubits))
    reverse_mapping_dict = list(range(self.num_qubits))

    for lq, target_phys in lq_to_phys.items():
        current_phys = mapping_dict[lq]
        if current_phys == target_phys:
            continue
        displaced_lq = reverse_mapping_dict[target_phys]

        mapping_dict[lq]           = target_phys
        mapping_dict[displaced_lq] = current_phys
        reverse_mapping_dict[target_phys]  = lq
        reverse_mapping_dict[current_phys] = displaced_lq

    self.mapping_dict         = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)