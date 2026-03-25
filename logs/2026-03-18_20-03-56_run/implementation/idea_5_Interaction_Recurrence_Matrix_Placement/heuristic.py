def init_mapping(self):
    """
    Interaction Recurrence Matrix Placement (IRMP).

    Borrows recurrence analysis from nonlinear dynamics: the "recurrence"
    of a qubit pair (q1, q2) is the sum of interaction occurrences weighted
    by an exponential recency factor — interactions in later circuit layers
    receive a higher bonus. Pairs with high recurrence score are likely to
    need sustained proximity throughout execution and are anchored on
    adjacent hardware nodes.

    Algorithm outline:
      1. ASAP greedy layer assignment (qubit-based, no DAG required).
      2. Recurrence matrix R[q1][q2] = Σ exp(α · L / L_max) for each
         2-qubit gate on (q1,q2) at circuit layer L.
      3. Anchor: highest-recurrence pair → most-central hardware node pair.
      4. Greedy recurrence-driven BFS expansion for remaining qubits.
      5. Encode final assignment as a strict 1-to-1 permutation.
    """
    from collections import defaultdict

    # --- Step 1: Collect logical qubits and assign circuit layers (ASAP) ---
    logical_qubit_set = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)

    # Fallback: trivial identity if no gates
    if not logical_qubit_set:
        self.mapping_dict = list(range(self.num_qubits))
        self.reverse_mapping_dict = list(range(self.num_qubits))
        if self.use_isl:
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return

    # ASAP layer assignment: process gates in sorted key order
    qubit_layer = defaultdict(int)  # qubit -> next available layer
    gate_layer = {}
    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        if not qubits:
            gate_layer[gate] = 0
            continue
        layer = max(qubit_layer[q] for q in qubits)
        gate_layer[gate] = layer
        for q in qubits:
            qubit_layer[q] = layer + 1

    L_max = max(gate_layer.values()) if gate_layer else 1

    # --- Step 2: Build Interaction Recurrence Matrix ---
    # R[q1][q2] = Σ exp(α · L / L_max)  for each 2q gate on (q1,q2) at layer L.
    # α > 0 gives an exponential recency bonus: later occurrences matter more.
    alpha = 2.0
    recurrence = defaultdict(lambda: defaultdict(float))

    for gate in sorted(self.access.keys()):
        qubits = self.access[gate]
        if len(qubits) == 2:
            q1, q2 = qubits[0], qubits[1]
            L = gate_layer[gate]
            w = math.exp(alpha * L / L_max)
            recurrence[q1][q2] += w
            recurrence[q2][q1] += w

    # Weighted recurrence degree per qubit
    recurrence_degree = defaultdict(float)
    for q1, neighbors in recurrence.items():
        recurrence_degree[q1] = sum(neighbors.values())

    logical_qubits = sorted(logical_qubit_set)
    physical_qubits = sorted(self.backend.keys())

    # --- Step 3: Identify anchor pair (highest recurrence score) ---
    best_pair = None
    best_score = -1.0
    for q1, neighbors in recurrence.items():
        for q2, w in neighbors.items():
            if q1 < q2 and w > best_score:
                best_score = w
                best_pair = (q1, q2)

    # Most-central physical qubit: minimum mean BFS distance to all others
    def mean_bfs_dist(p):
        finite = [
            self.distance_matrix[p][o]
            for o in physical_qubits
            if o != p and self.distance_matrix[p][o] != float('inf')
        ]
        return sum(finite) / len(finite) if finite else float('inf')

    anchor_physical = min(physical_qubits, key=mean_bfs_dist)

    lq_to_phys = {}
    placed_phys = set()

    if best_pair is not None:
        # Anchor the first qubit of the highest-recurrence pair on the
        # most-central physical node.
        anchor_lq, second_lq = best_pair
        lq_to_phys[anchor_lq] = anchor_physical
        placed_phys.add(anchor_physical)

        # Place the second qubit on the best adjacent hardware node
        # (adjacent = direct hardware edge exists → distance = 1).
        adjacent = [nb for nb in self.backend[anchor_physical]
                    if nb not in placed_phys]
        if adjacent:
            second_physical = min(adjacent, key=mean_bfs_dist)
        else:
            remaining = [p for p in physical_qubits if p not in placed_phys]
            second_physical = min(
                remaining,
                key=lambda p: self.distance_matrix[anchor_physical][p]
            )
        lq_to_phys[second_lq] = second_physical
        placed_phys.add(second_physical)
    else:
        # No 2-qubit interactions: anchor highest-degree qubit centrally
        anchor_lq = max(logical_qubits, key=lambda q: recurrence_degree[q])
        lq_to_phys[anchor_lq] = anchor_physical
        placed_phys.add(anchor_physical)

    # --- Step 4: Recurrence-driven greedy BFS expansion ---
    unplaced = [lq for lq in logical_qubits if lq not in lq_to_phys]

    while unplaced:
        # Next logical qubit: highest total recurrence weight to already-placed qubits
        next_lq = max(
            unplaced,
            key=lambda lq: sum(recurrence[lq].get(placed_lq, 0.0)
                               for placed_lq in lq_to_phys)
        )

        # Prefer hardware-adjacent candidates to stay topologically close
        candidates = list({
            nb
            for phys in placed_phys
            for nb in self.backend[phys]
            if nb not in placed_phys
        })
        if not candidates:
            candidates = [p for p in physical_qubits if p not in placed_phys]
        if not candidates:
            break

        # Cost: recurrence-weighted sum of BFS distances to placed partners
        def placement_cost(phys_c, _lq=next_lq):
            total = 0.0
            for placed_lq, placed_phys_q in lq_to_phys.items():
                w = recurrence[_lq].get(placed_lq, 0.0)
                if w > 0.0:
                    d = self.distance_matrix[phys_c][placed_phys_q]
                    total += w * (d if d != float('inf') else 1e9)
            return total

        best_phys = min(candidates, key=placement_cost)
        lq_to_phys[next_lq] = best_phys
        placed_phys.add(best_phys)
        unplaced.remove(next_lq)

    # --- Step 5: Fill remaining isolated logical qubits ---
    # Prefer high-degree hardware nodes for qubits with no interaction partners
    remaining_phys = sorted(
        [p for p in physical_qubits if p not in placed_phys],
        key=lambda p: len(self.backend[p]),
        reverse=True
    )
    for lq, phys in zip(unplaced, remaining_phys):
        lq_to_phys[lq] = phys

    # --- Step 6: Encode as strict 1-to-1 permutation via in-place swaps ---
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