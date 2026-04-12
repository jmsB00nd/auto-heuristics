def init_mapping(self):
    from collections import defaultdict, deque

    # --- Step 1: Build a per-qubit dependency DAG for layering ---
    last_gate_on_qubit = {}
    successors_local = defaultdict(set)
    predecessors_local = defaultdict(set)

    all_gates = sorted(self.access.keys())

    for gate in all_gates:
        for q in self.access[gate]:
            if q in last_gate_on_qubit:
                pred = last_gate_on_qubit[q]
                successors_local[pred].add(gate)
                predecessors_local[gate].add(pred)
            last_gate_on_qubit[q] = gate
        predecessors_local.setdefault(gate, set())
        successors_local.setdefault(gate, set())

    # Collect logical qubits and physical qubits
    logical_qubit_set = set()
    for qubits in self.access.values():
        logical_qubit_set.update(qubits)

    if not logical_qubit_set:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    physical_qubits = sorted(self.backend.keys())
    phys_degree = {p: len(self.backend[p]) for p in physical_qubits}

    # --- Step 2: Build wavefront layers (BFS topological order) ---
    pending_count = {gate: len(predecessors_local[gate]) for gate in all_gates}
    layers = []
    current_front = [g for g in all_gates if pending_count[g] == 0]

    while current_front:
        layers.append(current_front)
        next_front = []
        for gate in current_front:
            for succ in successors_local[gate]:
                pending_count[succ] -= 1
                if pending_count[succ] == 0:
                    next_front.append(succ)
        current_front = next_front

    # --- Step 3: Build per-logical-qubit interaction adjacency ---
    interaction_adj = defaultdict(lambda: defaultdict(float))
    for gate in all_gates:
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits
            interaction_adj[q1][q2] += 1.0
            interaction_adj[q2][q1] += 1.0

    # --- Step 4: Incremental placement with lookahead ---
    LOOKAHEAD_LAYERS = 5
    lq_to_phys = {}
    placed_phys = set()

    # Precompute centrality (mean distance to all others) for tie-breaking
    def _centrality(p):
        finite = [
            self.distance_matrix[p][o]
            for o in physical_qubits
            if p != o and self.distance_matrix[p][o] != float('inf')
        ]
        return sum(finite) / len(finite) if finite else float('inf')

    phys_centrality = {p: _centrality(p) for p in physical_qubits}

    def _score_candidate(candidate_p, logical_q, layer_idx):
        """
        Score a candidate physical qubit for placing logical_q.
        Lower is better. Considers:
        - Distance to already-placed interacting partners
        - Lookahead: predicted future interactions from upcoming layers
        - Centrality tie-breaking when no context exists
        """
        score = 0.0
        has_context = False

        # Cost from already-placed interaction partners
        for partner_lq, weight in interaction_adj[logical_q].items():
            if partner_lq in lq_to_phys:
                has_context = True
                partner_phys = lq_to_phys[partner_lq]
                score += weight * self.distance_matrix[candidate_p][partner_phys]

        # Lookahead: scan future layers for 2-qubit gates involving logical_q
        end_layer = min(layer_idx + LOOKAHEAD_LAYERS + 1, len(layers))
        for future_l in range(layer_idx + 1, end_layer):
            depth = future_l - layer_idx
            decay = 1.0 / (depth + 1)
            for fg in layers[future_l]:
                fq = self.access[fg]
                if len(fq) != 2:
                    continue
                fq1, fq2 = fq
                partner = None
                if fq1 == logical_q:
                    partner = fq2
                elif fq2 == logical_q:
                    partner = fq1
                if partner is not None and partner in lq_to_phys:
                    has_context = True
                    score += decay * self.distance_matrix[candidate_p][lq_to_phys[partner]]

        # If no placed neighbors give context, prefer central high-degree nodes
        if not has_context:
            score = phys_centrality[candidate_p] - 0.01 * phys_degree[candidate_p]

        return score

    def _place(logical_q, layer_idx, preferred_neighbor_of=None):
        """Place a logical qubit onto the best available physical qubit."""
        if logical_q in lq_to_phys:
            return lq_to_phys[logical_q]

        # Build candidate list: prefer neighbors of anchor if given
        if preferred_neighbor_of is not None:
            candidates = [
                p for p in self.backend[preferred_neighbor_of]
                if p not in placed_phys
            ]
        else:
            candidates = []

        # Fall back to all free physical qubits if no adjacent candidates
        if not candidates:
            candidates = [p for p in physical_qubits if p not in placed_phys]

        if not candidates:
            return None

        best_p = min(candidates, key=lambda p: _score_candidate(p, logical_q, layer_idx))
        lq_to_phys[logical_q] = best_p
        placed_phys.add(best_p)
        return best_p

    # --- Step 5: Process layers as wavefronts ---
    for layer_idx, layer in enumerate(layers):
        for gate in layer:
            qubits = self.access[gate]

            if len(qubits) == 1:
                _place(qubits[0], layer_idx)

            elif len(qubits) == 2:
                lq1, lq2 = qubits
                p1_exists = lq1 in lq_to_phys
                p2_exists = lq2 in lq_to_phys

                if p1_exists and p2_exists:
                    continue

                elif p1_exists:
                    _place(lq2, layer_idx, preferred_neighbor_of=lq_to_phys[lq1])

                elif p2_exists:
                    _place(lq1, layer_idx, preferred_neighbor_of=lq_to_phys[lq2])

                else:
                    # Neither placed: find best free adjacent hardware edge
                    best_score = float('inf')
                    best_p1, best_p2 = None, None

                    for p1 in physical_qubits:
                        if p1 in placed_phys:
                            continue
                        for p2 in self.backend[p1]:
                            if p2 in placed_phys or p2 <= p1:
                                continue
                            # Try both orientations
                            for pa, pb in [(p1, p2), (p2, p1)]:
                                s = _score_candidate(pa, lq1, layer_idx) + \
                                    _score_candidate(pb, lq2, layer_idx)
                                if s < best_score:
                                    best_score = s
                                    best_p1, best_p2 = pa, pb

                    if best_p1 is not None:
                        lq_to_phys[lq1] = best_p1
                        lq_to_phys[lq2] = best_p2
                        placed_phys.add(best_p1)
                        placed_phys.add(best_p2)
                    else:
                        # No free adjacent edge; place individually
                        phys1 = _place(lq1, layer_idx)
                        if phys1 is not None:
                            _place(lq2, layer_idx, preferred_neighbor_of=phys1)
                        else:
                            _place(lq2, layer_idx)

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