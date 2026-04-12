def init_mapping(self):
    import math
    import random
    from collections import defaultdict, deque

    num_qubits = self.num_qubits

    # ── Identify logical qubits actually used in the circuit ──
    logical_qubits_used = set()
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubits_used.add(q)
    logical_qubits_used = sorted(logical_qubits_used)

    # ── Phase 1: Decompose heavy-hex into chains and junctions ──
    degree = {}
    for node in self.backend:
        degree[node] = len(self.backend[node])

    junction_nodes = set()
    chain_interior_nodes = set()
    leaf_nodes = set()
    for node in self.backend:
        d = degree.get(node, 0)
        if d >= 3:
            junction_nodes.add(node)
        elif d == 1:
            leaf_nodes.add(node)
        else:
            chain_interior_nodes.add(node)

    # Extract hardware chains: paths of degree-2 (and leaf) nodes between junctions
    hardware_chains = []
    visited_chain = set()

    def extract_chain_from(start, prev):
        """Follow a chain from start (coming from prev) until hitting a junction or dead end."""
        chain = [start]
        visited_chain.add(start)
        current = start
        predecessor = prev
        while True:
            next_nodes = [n for n in self.backend[current] if n != predecessor]
            if not next_nodes:
                break
            # Pick the non-junction neighbor to continue chain
            found_next = False
            for n in next_nodes:
                if n in junction_nodes:
                    # Chain ends at this junction (don't include junction in chain)
                    chain.append(n)  # Mark endpoint as junction
                    return chain, n
                if n not in visited_chain:
                    visited_chain.add(n)
                    chain.append(n)
                    predecessor = current
                    current = n
                    found_next = True
                    break
            if not found_next:
                break
        return chain, None

    # Start chains from each junction's neighbors that are non-junctions
    for jnode in sorted(junction_nodes):
        for neighbor in self.backend[jnode]:
            if neighbor not in junction_nodes and neighbor not in visited_chain:
                chain, end_junction = extract_chain_from(neighbor, jnode)
                # A hardware chain: sequence of non-junction nodes, with junction endpoints
                hw_chain = {
                    'nodes': [n for n in chain if n not in junction_nodes],
                    'start_junction': jnode,
                    'end_junction': end_junction,
                    'full_path': [jnode] + [n for n in chain if n not in junction_nodes] + ([end_junction] if end_junction and end_junction != jnode else [])
                }
                if hw_chain['nodes']:
                    hardware_chains.append(hw_chain)

    # Also pick up chains starting from leaf nodes not yet visited
    for lnode in sorted(leaf_nodes):
        if lnode not in visited_chain:
            visited_chain.add(lnode)
            chain = [lnode]
            current = lnode
            predecessor = None
            while True:
                next_nodes = [n for n in self.backend[current] if n != predecessor]
                found = False
                for n in next_nodes:
                    if n in junction_nodes:
                        chain.append(n)
                        hw_chain = {
                            'nodes': [x for x in chain if x not in junction_nodes],
                            'start_junction': None,
                            'end_junction': n,
                            'full_path': [x for x in chain if x not in junction_nodes] + [n]
                        }
                        if hw_chain['nodes']:
                            hardware_chains.append(hw_chain)
                        found = True
                        break
                    elif n not in visited_chain:
                        visited_chain.add(n)
                        chain.append(n)
                        predecessor = current
                        current = n
                        found = True
                        break
                if not found or current in junction_nodes:
                    break

    # If no chains found (non heavy-hex topology), treat as flat
    if not hardware_chains and not junction_nodes:
        # Fallback: just treat all nodes as one big pool
        all_physical = sorted(self.backend.keys())
        hardware_chains = [{'nodes': all_physical, 'start_junction': None, 'end_junction': None, 'full_path': all_physical}]

    # Collect all placeable positions: chain interiors + junctions
    all_chain_positions = set()
    for hc in hardware_chains:
        for n in hc['nodes']:
            all_chain_positions.add(n)

    # ── Phase 2: Build interaction graph ──
    interaction_weight = defaultdict(float)
    # Build simple DAG for critical path estimation
    gate_list = sorted(self.access.keys())
    two_q_gates = []
    for g in gate_list:
        if len(self.access[g]) == 2:
            two_q_gates.append(g)
            q1, q2 = self.access[g]
            pair = (min(q1, q2), max(q1, q2))
            interaction_weight[pair] += 1.0

    # Build interaction adjacency
    interaction_adj = defaultdict(set)
    interaction_edge_weight = {}
    for (q1, q2), w in interaction_weight.items():
        interaction_adj[q1].add(q2)
        interaction_adj[q2].add(q1)
        interaction_edge_weight[(q1, q2)] = w
        interaction_edge_weight[(q2, q1)] = w

    # ── Phase 2b: Extract interaction chains via greedy heaviest-path cover ──
    interaction_chains = []
    covered = set()

    def get_heaviest_path():
        """Greedily extract the heaviest-weight path from uncovered qubits."""
        best_path = []
        best_weight = -1

        uncovered_qubits = [q for q in logical_qubits_used if q not in covered]
        if not uncovered_qubits:
            return []

        for start_q in uncovered_qubits:
            # Greedy extend from start_q
            path = [start_q]
            path_weight = 0
            used_in_path = {start_q}

            current = start_q
            while True:
                best_next = None
                best_w = 0
                for neighbor in interaction_adj.get(current, set()):
                    if neighbor not in covered and neighbor not in used_in_path:
                        w = interaction_edge_weight.get((current, neighbor), 0)
                        if w > best_w:
                            best_w = w
                            best_next = neighbor
                if best_next is None:
                    break
                path.append(best_next)
                path_weight += best_w
                used_in_path.add(best_next)
                current = best_next

            if path_weight > best_weight or (path_weight == best_weight and len(path) > len(best_path)):
                best_path = path
                best_weight = path_weight

        return best_path

    # Extract interaction chains
    while len(covered) < len(logical_qubits_used):
        path = get_heaviest_path()
        if not path:
            # Cover remaining singletons
            for q in logical_qubits_used:
                if q not in covered:
                    interaction_chains.append([q])
                    covered.add(q)
            break
        interaction_chains.append(path)
        for q in path:
            covered.add(q)

    # ── Phase 3: Assign interaction chains to hardware chains ──
    # Sort hardware chains by length (descending) and interaction chains by length (descending)
    hw_chains_sorted = sorted(range(len(hardware_chains)), key=lambda i: len(hardware_chains[i]['nodes']), reverse=True)
    int_chains_sorted = sorted(range(len(interaction_chains)), key=lambda i: len(interaction_chains[i]), reverse=True)

    # Simple greedy assignment: biggest interaction chain -> biggest hardware chain
    chain_assignment = {}  # interaction_chain_idx -> hardware_chain_idx
    used_hw = set()

    # For Hungarian-like assignment, build cost matrix for chains that fit
    # Use simplified greedy for efficiency
    remaining_int_chains = list(int_chains_sorted)
    remaining_hw_chains = list(hw_chains_sorted)

    for ic_idx in remaining_int_chains[:]:
        ic_len = len(interaction_chains[ic_idx])
        best_hw = None
        best_cost = float('inf')

        for hc_idx in remaining_hw_chains:
            if hc_idx in used_hw:
                continue
            hc_len = len(hardware_chains[hc_idx]['nodes'])
            if hc_len >= ic_len:
                # Cost: wasted space + centrality penalty
                cost = hc_len - ic_len
                if cost < best_cost:
                    best_cost = cost
                    best_hw = hc_idx

        if best_hw is not None:
            chain_assignment[ic_idx] = best_hw
            used_hw.add(best_hw)

    # ── Phase 4: Place qubits within matched chains ──
    mapping_dict = list(range(num_qubits))  # identity initially
    reverse_mapping_dict = list(range(num_qubits))
    placed_logical = set()
    used_physical = set()

    def place(logical_q, physical_q):
        """Place logical_q at physical_q, maintaining bijection."""
        nonlocal mapping_dict, reverse_mapping_dict
        old_phys = mapping_dict[logical_q]
        old_log = reverse_mapping_dict[physical_q]

        mapping_dict[logical_q] = physical_q
        mapping_dict[old_log] = old_phys
        reverse_mapping_dict[physical_q] = logical_q
        reverse_mapping_dict[old_phys] = old_log

        placed_logical.add(logical_q)
        used_physical.add(physical_q)

    for ic_idx, hc_idx in chain_assignment.items():
        ic = interaction_chains[ic_idx]
        hc_nodes = hardware_chains[hc_idx]['nodes']

        # For small chains, try both orderings (forward/reverse) and pick best
        # For larger chains, use greedy ordering along the chain's physical path
        if len(ic) <= 8:
            # Try forward and reverse, pick best intra-chain cost
            best_order = ic
            best_cost = float('inf')

            for order in [ic, list(reversed(ic))]:
                cost = 0
                for i in range(len(order) - 1):
                    q1, q2 = order[i], order[i + 1]
                    p1, p2 = hc_nodes[i], hc_nodes[i + 1] if i + 1 < len(hc_nodes) else hc_nodes[-1]
                    w = interaction_edge_weight.get((q1, q2), 0) + interaction_edge_weight.get((q2, q1), 0)
                    cost += w * self.distance_matrix[p1][p2]
                if cost < best_cost:
                    best_cost = cost
                    best_order = order

            for i, lq in enumerate(best_order):
                if i < len(hc_nodes):
                    place(lq, hc_nodes[i])
        else:
            # Greedy: place in order
            for i, lq in enumerate(ic):
                if i < len(hc_nodes):
                    place(lq, hc_nodes[i])

    # ── Phase 4b: Place interaction chains that didn't fit in hardware chains ──
    # Place remaining unassigned interaction chains greedily
    unassigned_ic = [i for i in range(len(interaction_chains)) if i not in chain_assignment]

    for ic_idx in unassigned_ic:
        ic = interaction_chains[ic_idx]
        for lq in ic:
            if lq in placed_logical:
                continue
            # Find best available physical qubit: minimize distance to already-placed neighbors
            best_pq = None
            best_score = float('inf')
            neighbors_placed = []
            for nq in interaction_adj.get(lq, set()):
                if nq in placed_logical:
                    neighbors_placed.append(mapping_dict[nq])

            candidates = [p for p in self.backend if p not in used_physical]
            if not candidates:
                candidates = [p for p in self.backend]

            for pq in candidates:
                if pq in used_physical:
                    continue
                score = 0
                for np_phys in neighbors_placed:
                    score += self.distance_matrix[pq][np_phys]
                if not neighbors_placed:
                    # Prefer central nodes
                    score = sum(self.distance_matrix[pq][other] for other in self.backend if other != pq)
                if score < best_score:
                    best_score = score
                    best_pq = pq

            if best_pq is not None:
                place(lq, best_pq)

    # Place any remaining logical qubits that weren't in any interaction chain
    for lq in logical_qubits_used:
        if lq not in placed_logical:
            for pq in self.backend:
                if pq not in used_physical:
                    place(lq, pq)
                    break

    # ── Phase 5: ILS + SA Refinement ──
    def compute_total_cost(m_dict):
        cost = 0
        for g in two_q_gates:
            q1, q2 = self.access[g]
            p1, p2 = m_dict[q1], m_dict[q2]
            cost += self.distance_matrix[p1][p2]
        return cost

    current_cost = compute_total_cost(mapping_dict)
    best_mapping = list(mapping_dict)
    best_reverse = list(reverse_mapping_dict)
    best_cost = current_cost

    if len(logical_qubits_used) > 1 and two_q_gates:
        temperature = max(1.0, current_cost * 0.1)
        cooling_rate = 0.995
        max_iterations = min(8000, len(logical_qubits_used) * 100)
        lq_list = list(logical_qubits_used)

        for iteration in range(max_iterations):
            # Pick two random logical qubits and swap their physical assignments
            i1, i2 = random.sample(range(len(lq_list)), 2)
            lq1, lq2 = lq_list[i1], lq_list[i2]
            pq1, pq2 = mapping_dict[lq1], mapping_dict[lq2]

            # Compute delta cost (only affected gates)
            delta = 0
            for g in two_q_gates:
                qa, qb = self.access[g]
                old_pa, old_pb = mapping_dict[qa], mapping_dict[qb]
                # Compute new physical positions after swap
                new_pa = old_pa
                new_pb = old_pb
                if qa == lq1:
                    new_pa = pq2
                elif qa == lq2:
                    new_pa = pq1
                if qb == lq1:
                    new_pb = pq2
                elif qb == lq2:
                    new_pb = pq1
                if new_pa != old_pa or new_pb != old_pb:
                    delta += self.distance_matrix[new_pa][new_pb] - self.distance_matrix[old_pa][old_pb]

            if delta < 0 or (temperature > 0.01 and random.random() < math.exp(-delta / max(temperature, 1e-10))):
                # Accept swap
                mapping_dict[lq1] = pq2
                mapping_dict[lq2] = pq1
                reverse_mapping_dict[pq1] = lq2
                reverse_mapping_dict[pq2] = lq1
                current_cost += delta

                if current_cost < best_cost:
                    best_cost = current_cost
                    best_mapping = list(mapping_dict)
                    best_reverse = list(reverse_mapping_dict)

            temperature *= cooling_rate

        # ILS: perturbation + re-anneal
        for ils_round in range(3):
            # Perturbation: random swaps
            mapping_dict = list(best_mapping)
            reverse_mapping_dict = list(best_reverse)
            current_cost = best_cost

            num_perturb = max(2, len(lq_list) // 5)
            for _ in range(num_perturb):
                i1, i2 = random.sample(range(len(lq_list)), 2)
                lq1, lq2 = lq_list[i1], lq_list[i2]
                pq1, pq2 = mapping_dict[lq1], mapping_dict[lq2]
                mapping_dict[lq1] = pq2
                mapping_dict[lq2] = pq1
                reverse_mapping_dict[pq1] = lq2
                reverse_mapping_dict[pq2] = lq1

            current_cost = compute_total_cost(mapping_dict)
            temperature = max(0.5, best_cost * 0.05)

            for iteration in range(max_iterations // 2):
                i1, i2 = random.sample(range(len(lq_list)), 2)
                lq1, lq2 = lq_list[i1], lq_list[i2]
                pq1, pq2 = mapping_dict[lq1], mapping_dict[lq2]

                delta = 0
                for g in two_q_gates:
                    qa, qb = self.access[g]
                    old_pa, old_pb = mapping_dict[qa], mapping_dict[qb]
                    new_pa, new_pb = old_pa, old_pb
                    if qa == lq1: new_pa = pq2
                    elif qa == lq2: new_pa = pq1
                    if qb == lq1: new_pb = pq2
                    elif qb == lq2: new_pb = pq1
                    if new_pa != old_pa or new_pb != old_pb:
                        delta += self.distance_matrix[new_pa][new_pb] - self.distance_matrix[old_pa][old_pb]

                if delta < 0 or (temperature > 0.01 and random.random() < math.exp(-delta / max(temperature, 1e-10))):
                    mapping_dict[lq1] = pq2
                    mapping_dict[lq2] = pq1
                    reverse_mapping_dict[pq1] = lq2
                    reverse_mapping_dict[pq2] = lq1
                    current_cost += delta

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_mapping = list(mapping_dict)
                        best_reverse = list(reverse_mapping_dict)

                temperature *= cooling_rate

    self.mapping_dict = best_mapping
    self.reverse_mapping_dict = best_reverse

    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)