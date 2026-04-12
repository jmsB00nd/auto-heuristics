def init_mapping(self):
    from collections import defaultdict
    
    num_q = self.num_qubits
    BEAM_WIDTH = 10
    
    # Step 1: Build interaction graph from self.access
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)
    logical_qubits_set = set()
    
    for gate, qubits in self.access.items():
        if len(qubits) == 2:
            q1, q2 = qubits
            logical_qubits_set.add(q1)
            logical_qubits_set.add(q2)
            key = (min(q1, q2), max(q1, q2))
            interaction_weight[key] += 1.0
            logical_degree[q1] += 1.0
            logical_degree[q2] += 1.0
        elif len(qubits) == 1:
            logical_qubits_set.add(qubits[0])
    
    # Step 2: Order logical qubits by interaction degree descending
    logical_qubits_ordered = sorted(logical_qubits_set, key=lambda q: logical_degree.get(q, 0), reverse=True)
    
    # Build per-qubit interaction list for fast lookup
    qubit_interactions = defaultdict(list)  # qubit -> [(other_qubit, weight)]
    for (q1, q2), w in interaction_weight.items():
        qubit_interactions[q1].append((q2, w))
        qubit_interactions[q2].append((q1, w))
    
    # Precompute average distance to neighbors for each physical qubit
    physical_qubits = sorted(self.backend.keys())
    all_physical = set(range(num_q))
    
    # Step 3: Beam search
    # Each candidate: (score, mapping_partial, reverse_partial, placed_logical, used_physical)
    # Use tuples for immutability; store as dicts for speed
    
    initial_candidate = {
        'score': 0.0,
        'mapping': {},       # logical -> physical
        'reverse': {},       # physical -> logical
        'placed': set(),
        'used_physical': set()
    }
    beam = [initial_candidate]
    
    for lq in logical_qubits_ordered:
        next_beam = []
        
        for cand in beam:
            placed = cand['placed']
            used_phys = cand['used_physical']
            mapping = cand['mapping']
            base_score = cand['score']
            
            # Compute incremental score for placing lq at each free physical qubit
            # Interactions of lq with already-placed qubits
            placed_interactions = [(oq, w) for oq, w in qubit_interactions[lq] if oq in placed]
            # Interactions of lq with not-yet-placed qubits (for penalty)
            unplaced_interactions = [(oq, w) for oq, w in qubit_interactions[lq] if oq not in placed and oq != lq]
            
            free_physical = [p for p in physical_qubits if p not in used_phys]
            
            for pq in free_physical:
                # Cost from fully-placed pairs
                placed_cost = 0.0
                for oq, w in placed_interactions:
                    p_other = mapping[oq]
                    placed_cost += w * self.distance_matrix[pq][p_other]
                
                # Penalty for partially-placed (one placed, one not)
                penalty = 0.0
                if unplaced_interactions:
                    # avg distance to nearest free neighbor of pq
                    neighbors_of_pq = self.backend.get(pq, [])
                    free_neighbors = [n for n in neighbors_of_pq if n not in used_phys and n != pq]
                    if free_neighbors:
                        avg_nearest = sum(self.distance_matrix[pq][fn] for fn in free_neighbors) / len(free_neighbors)
                    else:
                        # No free neighbors, use avg distance to all free positions
                        other_free = [p for p in free_physical if p != pq]
                        if other_free:
                            avg_nearest = sum(self.distance_matrix[pq][p] for p in other_free) / len(other_free)
                        else:
                            avg_nearest = 0.0
                    
                    for oq, w in unplaced_interactions:
                        penalty += w * avg_nearest
                
                new_score = base_score + placed_cost + penalty
                
                next_beam.append({
                    'score': new_score,
                    'mapping': {**mapping, lq: pq},
                    'reverse': {**cand['reverse'], pq: lq},
                    'placed': placed | {lq},
                    'used_physical': used_phys | {pq}
                })
        
        # Keep top B candidates
        next_beam.sort(key=lambda c: c['score'])
        beam = next_beam[:BEAM_WIDTH]
    
    # Step 4: Best candidate
    best = beam[0]
    
    # Step 5: Build full mapping (fill unmapped qubits)
    mapping_dict = [-1] * num_q
    reverse_mapping_dict = [-1] * num_q
    
    for lq, pq in best['mapping'].items():
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq
    
    unmapped_logical = [q for q in range(num_q) if mapping_dict[q] == -1]
    free_physical = [q for q in range(num_q) if reverse_mapping_dict[q] == -1]
    
    for lq, pq in zip(unmapped_logical, free_physical):
        mapping_dict[lq] = pq
        reverse_mapping_dict[pq] = lq
    
    self.mapping_dict = mapping_dict
    self.reverse_mapping_dict = reverse_mapping_dict
    
    if self.use_isl:
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)