def init_mapping(self):
    N = self.num_qubits
    mapping = [-1] * N
    reverse_mapping = [-1] * N
    used_phys = set()
    placed_logical = set()

    # Collect 2-qubit interactions and aggregate weights using the QIG.
    edge_weights = {}
    logical_qubits_in_access = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                logical_qubits_in_access.add(a)
                continue
            logical_qubits_in_access.add(a)
            logical_qubits_in_access.add(b)
            key = (a, b) if a < b else (b, a)
            try:
                w = self.qubit_interaction_graph[key[0]][key[1]]
            except Exception:
                w = 0
            if not w:
                edge_weights[key] = edge_weights.get(key, 0) + 1
            else:
                edge_weights[key] = w
        elif len(qubits) == 1:
            logical_qubits_in_access.add(qubits[0])

    # Sort edges heaviest-first (Kruskal-flavored sweep).
    sorted_edges = sorted(edge_weights.items(), key=lambda kv: -kv[1])

    def closest_free_physical(anchor_phys):
        best_p = -1
        best_d = float('inf')
        for p in range(N):
            if p in used_phys:
                continue
            d = self.distance_matrix[anchor_phys][p]
            if d < best_d:
                best_d = d
                best_p = p
        return best_p

    def most_central_free():
        best_p = -1
        best_score = -float('inf')
        for p in range(N):
            if p in used_phys:
                continue
            s = self.physical_centrality.get(p, 0.0) if isinstance(self.physical_centrality, dict) else self.physical_centrality[p]
            if s > best_score:
                best_score = s
                best_p = p
        if best_p == -1:
            for p in range(N):
                if p not in used_phys:
                    return p
        return best_p

    def assign(logical_q, phys_q):
        if logical_q < 0 or logical_q >= N or phys_q < 0 or phys_q >= N:
            return False
        if logical_q in placed_logical or phys_q in used_phys:
            return False
        mapping[logical_q] = phys_q
        reverse_mapping[phys_q] = logical_q
        used_phys.add(phys_q)
        placed_logical.add(logical_q)
        return True

    # Process edges heaviest -> lightest.
    for (u, v), _w in sorted_edges:
        u_in = u in placed_logical
        v_in = v in placed_logical
        if u_in and v_in:
            continue
        if not u_in and not v_in:
            seed = most_central_free()
            if seed == -1:
                break
            if not assign(u, seed):
                continue
            partner = closest_free_physical(mapping[u])
            if partner != -1:
                assign(v, partner)
        elif u_in and not v_in:
            partner = closest_free_physical(mapping[u])
            if partner != -1:
                assign(v, partner)
        else:
            partner = closest_free_physical(mapping[v])
            if partner != -1:
                assign(u, partner)

    # Back-fill any remaining logical qubits seen in access onto most central free physicals.
    remaining_logicals = [L for L in logical_qubits_in_access if L not in placed_logical and 0 <= L < N]
    remaining_logicals.sort(key=lambda L: -self.logical_activity.get(L, 0) if hasattr(self.logical_activity, 'get') else 0)
    for L in remaining_logicals:
        p = most_central_free()
        if p == -1:
            break
        assign(L, p)

    # Pad remaining slots with identity-style assignment to keep lists fully populated and injective.
    free_phys = [p for p in range(N) if p not in used_phys]
    free_logical = [L for L in range(N) if L not in placed_logical]
    # Prefer identity where possible.
    free_logical_set = set(free_logical)
    free_phys_set = set(free_phys)
    for L in list(free_logical):
        if L in free_phys_set:
            mapping[L] = L
            reverse_mapping[L] = L
            used_phys.add(L)
            placed_logical.add(L)
            free_phys_set.discard(L)
            free_logical_set.discard(L)
    remaining_free_logical = sorted(free_logical_set)
    remaining_free_phys = sorted(free_phys_set)
    for L, p in zip(remaining_free_logical, remaining_free_phys):
        mapping[L] = p
        reverse_mapping[p] = L

    self.mapping_dict = mapping
    self.reverse_mapping_dict = reverse_mapping

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)