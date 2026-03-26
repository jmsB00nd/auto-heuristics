def init_mapping(self):
    # Step 1: Collect logical qubits and build interaction weight matrix
    logical_qubit_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)

    logical_qubits = sorted(logical_qubit_set)
    num_logical = len(logical_qubits)
    physical_nodes = sorted(self.backend.keys())

    # Build interaction weight matrix W[q1][q2] = total interaction count
    W = defaultdict(lambda: defaultdict(float))
    total_weight = defaultdict(float)
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            W[q1][q2] += 1.0
            W[q2][q1] += 1.0
            total_weight[q1] += 1.0
            total_weight[q2] += 1.0

    # Step 2: Initialize with trivial mapping
    self.mapping_dict = list(range(self.num_qubits))
    self.reverse_mapping_dict = list(range(self.num_qubits))

    if num_logical <= 1:
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    best_mapping = self.mapping_dict[:]
    best_reverse = self.reverse_mapping_dict[:]
    best_score = float('inf')

    def compute_total_weighted_distance(mapping):
        score = 0.0
        for gate, qubits in self.access.items():
            if len(qubits) == 2:
                q1, q2 = qubits[0], qubits[1]
                score += self.distance_matrix[mapping[q1]][mapping[q2]]
        return score

    # Step 3: Iterative gravitational relaxation
    T = 20  # number of iterations

    for iteration in range(T):
        # 3a: For each logical qubit, compute ideal physical position
        ideal_position = {}
        for q in logical_qubits:
            best_p = self.mapping_dict[q]
            best_cost = float('inf')
            for p in physical_nodes:
                cost = 0.0
                for j in logical_qubits:
                    if j == q:
                        continue
                    w = W[q][j]
                    if w > 0:
                        cost += w * self.distance_matrix[p][self.mapping_dict[j]]
                if cost < best_cost:
                    best_cost = cost
                    best_p = p
            ideal_position[q] = best_p

        # 3b: Resolve conflicts - highest total interaction weight wins
        # Group qubits by their desired position
        position_candidates = defaultdict(list)
        for q in logical_qubits:
            position_candidates[ideal_position[q]].append(q)

        assigned = {}  # logical -> physical
        occupied = set()

        # For contested positions, the qubit with highest total weight keeps it
        for p, candidates in position_candidates.items():
            if len(candidates) == 1:
                assigned[candidates[0]] = p
                occupied.add(p)
            else:
                # Sort by total interaction weight descending
                candidates.sort(key=lambda q: total_weight[q], reverse=True)
                # Winner keeps the position
                assigned[candidates[0]] = p
                occupied.add(p)
                # 3c: Displaced qubits get nearest unoccupied physical qubit
                for displaced_q in candidates[1:]:
                    # Find nearest unoccupied physical qubit
                    best_dist = float('inf')
                    best_free = None
                    for fp in physical_nodes:
                        if fp not in occupied:
                            d = self.distance_matrix[p][fp]
                            if d < best_dist:
                                best_dist = d
                                best_free = fp
                    if best_free is not None:
                        assigned[displaced_q] = best_free
                        occupied.add(best_free)

        # Apply assignments using swap-based approach to maintain bijectivity
        new_mapping = self.mapping_dict[:]
        new_reverse = self.reverse_mapping_dict[:]
        for lq, target_phys in assigned.items():
            current_phys = new_mapping[lq]
            if current_phys == target_phys:
                continue
            displaced_lq = new_reverse[target_phys]
            new_mapping[lq] = target_phys
            new_mapping[displaced_lq] = current_phys
            new_reverse[target_phys] = lq
            new_reverse[current_phys] = displaced_lq

        self.mapping_dict = new_mapping
        self.reverse_mapping_dict = new_reverse

        # Step 4: Track best mapping
        score = compute_total_weighted_distance(self.mapping_dict)
        if score < best_score:
            best_score = score
            best_mapping = self.mapping_dict[:]
            best_reverse = self.reverse_mapping_dict[:]

    # Step 5: Return best mapping found
    self.mapping_dict = best_mapping
    self.reverse_mapping_dict = best_reverse

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)