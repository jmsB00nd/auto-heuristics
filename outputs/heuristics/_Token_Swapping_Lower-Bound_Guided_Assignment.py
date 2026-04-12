def init_mapping(self):
    from collections import defaultdict

    num_q = self.num_qubits
    dm_size = len(self.distance_matrix)

    def dist(p1, p2):
        if p1 < dm_size and p2 < dm_size:
            return self.distance_matrix[p1][p2]
        return float('inf')

    # ── Step 0: Extract 2-qubit gates and build interaction graph ────
    two_q_gates = []
    interaction_degree = defaultdict(int)

    for gate_id in sorted(self.access.keys()):
        qubits = self.access[gate_id]
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            two_q_gates.append((gate_id, q1, q2))
            interaction_degree[q1] += 1
            interaction_degree[q2] += 1

    if not two_q_gates:
        self.mapping_dict = list(range(num_q))
        self.reverse_mapping_dict = list(range(num_q))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ── Step 1: Identify first-layer gates for token-swap LB ────────
    first_layer_pairs = []
    used_in_layer = set()
    for gid, q1, q2 in two_q_gates:
        if q1 not in used_in_layer and q2 not in used_in_layer:
            first_layer_pairs.append((q1, q2))
            used_in_layer.add(q1)
            used_in_layer.add(q2)
        elif q1 in used_in_layer or q2 in used_in_layer:
            break

    # Build layers with decay weights for broader cost guidance
    layers = []
    used = set()
    current_layer = []
    for gid, q1, q2 in two_q_gates:
        if q1 in used or q2 in used:
            if current_layer:
                layers.append(current_layer)
            current_layer = [(q1, q2)]
            used = {q1, q2}
        else:
            current_layer.append((q1, q2))
            used.add(q1)
            used.add(q2)
    if current_layer:
        layers.append(current_layer)

    # Weighted gate pairs: first layers matter more
    weighted_pairs = []
    for li, layer in enumerate(layers[:5]):
        weight = 1.0 / (li + 1)
        for q1, q2 in layer:
            weighted_pairs.append((q1, q2, weight))

    # ── Step 2: Order logical qubits by interaction degree (desc) ────
    all_logical = set(range(num_q))
    involved_logical = set()
    for q1, q2, _ in weighted_pairs:
        involved_logical.add(q1)
        involved_logical.add(q2)

    sorted_involved = sorted(involved_logical,
                             key=lambda q: interaction_degree[q],
                             reverse=True)
    uninvolved = sorted(all_logical - involved_logical)
    qubit_order = sorted_involved + uninvolved

    # ── Step 3: Beam search over partial assignments ─────────────────
    BEAM_WIDTH = 32
    all_physical = set(range(num_q))

    def compute_cost(partial_mapping):
        """Sum of weighted pairwise distances for already-constrained gates."""
        cost = 0.0
        for q1, q2, w in weighted_pairs:
            p1 = partial_mapping.get(q1)
            p2 = partial_mapping.get(q2)
            if p1 is not None and p2 is not None:
                cost += w * dist(p1, p2)
        return cost

    # beam entries: (cost, partial_mapping_dict, placed_physical_set)
    beam = [(0.0, {}, set())]

    for logical_q in qubit_order:
        next_beam = []

        for cost, partial_map, placed_phys in beam:
            available_phys = all_physical - placed_phys
            candidates = list(available_phys)

            # Pre-filter: keep only top-B candidates by incremental cost
            if len(candidates) > BEAM_WIDTH and logical_q in involved_logical:
                scored = []
                for pq in candidates:
                    incr_cost = 0.0
                    for q1, q2, w in weighted_pairs:
                        partner = None
                        if q1 == logical_q:
                            partner = q2
                        elif q2 == logical_q:
                            partner = q1
                        if partner is not None:
                            pp = partial_map.get(partner)
                            if pp is not None:
                                incr_cost += w * dist(pq, pp)
                    scored.append((incr_cost, pq))
                scored.sort()
                candidates = [pq for _, pq in scored[:BEAM_WIDTH]]

            for pq in candidates:
                new_map = partial_map.copy()
                new_map[logical_q] = pq
                new_placed = placed_phys | {pq}
                new_cost = compute_cost(new_map)
                next_beam.append((new_cost, new_map, new_placed))

        # Keep top-B candidates
        next_beam.sort(key=lambda x: x[0])
        beam = next_beam[:BEAM_WIDTH]

    # ── Step 4: Token-swapping lower bound to pick best ──────────────
    def token_swap_lb(mapping_list):
        """LB on SWAPs for first-layer gates: each gate at distance d needs >= d-1 swaps."""
        total = 0.0
        for q1, q2 in first_layer_pairs:
            p1 = mapping_list[q1]
            p2 = mapping_list[q2]
            d = dist(p1, p2)
            if d > 1:
                total += d - 1
        return total

    best_score = float('inf')
    best_mapping = None

    for cost, partial_map, _ in beam:
        mapping_list = [0] * num_q
        for lq, pq in partial_map.items():
            mapping_list[lq] = pq

        tscore = cost + token_swap_lb(mapping_list)
        if tscore < best_score:
            best_score = tscore
            best_mapping = mapping_list

    # ── Step 5: Populate mapping dicts ───────────────────────────────
    self.mapping_dict = best_mapping[:]
    self.reverse_mapping_dict = [0] * num_q
    for logical, physical in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[physical] = logical

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)