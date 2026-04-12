def init_mapping(self):
    """
    Degree-Sequence Histogram Matching with Tabu-Search Refinement (DSHM-TS).

    Phase 1 – Sorted Degree Matching:
        Compute the weighted degree of each logical qubit (sum of all pairwise
        interaction counts with its partners in the circuit interaction graph).
        Compute the hardware degree of each physical qubit (number of direct
        hardware edges). Sort both lists in descending order and match them
        positionally: highest-weighted-degree logical qubit -> highest-hardware-
        degree physical qubit, second -> second, etc.

    Phase 2 – Tabu-Search Refinement:
        Starting from the degree-matched placement, run a tabu search over all
        pairwise logical-qubit position swaps. A move (lq_a <-> lq_b) is tabu
        for `tabu_tenure` iterations after it is applied, preventing immediate
        reversal and forcing neighbourhood diversification. Aspiration criterion:
        a tabu move is accepted if it produces a cost strictly below the global
        best. Terminates when the move budget or no-improvement patience is
        exhausted.
    """
    from collections import defaultdict

    # ── Build weighted circuit interaction graph ──────────────────────────
    interaction_weight = defaultdict(float)
    logical_qubit_set = set()

    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0

    # Trivial fallback for empty circuits
    if not logical_qubit_set:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # Weighted degree and neighbour adjacency for each logical qubit
    weighted_degree = defaultdict(float)
    interaction_neighbors = defaultdict(dict)
    for (q1, q2), w in interaction_weight.items():
        weighted_degree[q1] += w
        weighted_degree[q2] += w
        interaction_neighbors[q1][q2] = interaction_neighbors[q1].get(q2, 0) + w
        interaction_neighbors[q2][q1] = interaction_neighbors[q2].get(q1, 0) + w

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # ── Phase 1: Degree-Sequence Sorted Matching ─────────────────────────
    sorted_logical  = sorted(logical_qubits,  key=lambda q: weighted_degree[q],      reverse=True)
    sorted_physical = sorted(physical_qubits, key=lambda p: len(self.backend[p]),    reverse=True)

    lq_to_phys = {}
    for lq, ph in zip(sorted_logical, sorted_physical):
        lq_to_phys[lq] = ph

    # Remaining isolated logical qubits -> leftover physical qubits (same order)
    placed_phys = set(lq_to_phys.values())
    rem_lq   = [lq for lq in logical_qubits  if lq not in lq_to_phys]
    rem_phys = [p  for p  in sorted_physical if p  not in placed_phys]
    for lq, ph in zip(rem_lq, rem_phys):
        lq_to_phys[lq] = ph

    # Build strict bijective mapping via in-place transpositions on identity
    def build_mapping(assignment):
        m  = list(range(self.num_qubits))
        rm = list(range(self.num_qubits))
        for lq, target in assignment.items():
            current  = m[lq]
            if current == target:
                continue
            displaced = rm[target]
            m[lq]        = target;   m[displaced]  = current
            rm[target]   = lq;       rm[current]   = displaced
        return m, rm

    mapping_dict, reverse_mapping_dict = build_mapping(lq_to_phys)

    # ── Helpers ───────────────────────────────────────────────────────────
    def total_cost(m):
        cost = 0.0
        for (q1, q2), w in interaction_weight.items():
            d = self.distance_matrix[m[q1]][m[q2]]
            cost += w * (d if d != float('inf') else 1e9)
        return cost

    def delta_swap(lq_a, lq_b, m):
        """O(deg(a)+deg(b)) incremental cost change for swapping lq_a <-> lq_b."""
        p_a, p_b = m[lq_a], m[lq_b]
        delta = 0.0
        for lq_c, w in interaction_neighbors[lq_a].items():
            if lq_c != lq_b:
                p_c = m[lq_c]
                d_new = self.distance_matrix[p_b][p_c]
                d_old = self.distance_matrix[p_a][p_c]
                delta += w * ((d_new if d_new != float('inf') else 1e9) -
                              (d_old if d_old != float('inf') else 1e9))
        for lq_c, w in interaction_neighbors[lq_b].items():
            if lq_c != lq_a:
                p_c = m[lq_c]
                d_new = self.distance_matrix[p_a][p_c]
                d_old = self.distance_matrix[p_b][p_c]
                delta += w * ((d_new if d_new != float('inf') else 1e9) -
                              (d_old if d_old != float('inf') else 1e9))
        return delta

    def apply_swap(lq_a, lq_b, m, rm):
        p_a, p_b = m[lq_a], m[lq_b]
        m[lq_a],  m[lq_b]  = p_b, p_a
        rm[p_a],  rm[p_b]  = lq_b, lq_a

    # ── Phase 2: Tabu-Search Refinement ──────────────────────────────────
    active_lqs = sorted([lq for lq in logical_qubits if interaction_neighbors[lq]])
    n_active   = len(active_lqs)

    if n_active < 2:
        self.mapping_dict        = mapping_dict
        self.reverse_mapping_dict = reverse_mapping_dict
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    tabu_tenure      = max(5,  n_active // 4)
    max_moves        = max(60, n_active * n_active)
    no_improve_limit = max(25, n_active * 3)

    tabu_timestamps  = {}   # (lq_a, lq_b) -> move_num of last application

    current_cost = total_cost(mapping_dict)
    best_cost    = current_cost
    best_mapping = mapping_dict[:]
    best_reverse = reverse_mapping_dict[:]

    no_improve_count = 0
    move_num         = 0

    while move_num < max_moves and no_improve_count < no_improve_limit:
        candidate_delta = float('inf')
        candidate_pair  = None

        for i in range(n_active):
            for j in range(i + 1, n_active):
                lq_a, lq_b = active_lqs[i], active_lqs[j]
                key = (lq_a, lq_b)          # always min-first since list is sorted

                d               = delta_swap(lq_a, lq_b, mapping_dict)
                projected_cost  = current_cost + d
                is_tabu         = (key in tabu_timestamps and
                                   move_num - tabu_timestamps[key] < tabu_tenure)
                aspiration      = projected_cost < best_cost

                if (not is_tabu or aspiration) and d < candidate_delta:
                    candidate_delta = d
                    candidate_pair  = (lq_a, lq_b)

        if candidate_pair is None:
            break   # fully tabu neighbourhood, no aspiration — terminate

        lq_a, lq_b = candidate_pair
        apply_swap(lq_a, lq_b, mapping_dict, reverse_mapping_dict)
        tabu_timestamps[(lq_a, lq_b)] = move_num
        current_cost += candidate_delta

        if current_cost < best_cost:
            best_cost    = current_cost
            best_mapping = mapping_dict[:]
            best_reverse = reverse_mapping_dict[:]
            no_improve_count = 0
        else:
            no_improve_count += 1

        move_num += 1

    self.mapping_dict         = best_mapping
    self.reverse_mapping_dict = best_reverse

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)