def init_mapping(self):
    import random
    from collections import defaultdict, deque
    
    num_q = self.num_qubits
    physical_qubits = sorted(self.backend.keys())
    
    # --- Identify logical qubits used in the circuit ---
    logical_qubit_set = set()
    two_qubit_gates = []
    for gate, qubits in self.access.items():
        for q in qubits:
            logical_qubit_set.add(q)
        if len(qubits) == 2:
            two_qubit_gates.append(gate)
    
    logical_qubits = sorted(logical_qubit_set)
    n_logical = len(logical_qubits)
    
    # --- Build interaction weights between logical qubits ---
    interaction_weight = defaultdict(float)
    logical_degree = defaultdict(float)
    for g in two_qubit_gates:
        q1, q2 = self.access[g]
        key = (min(q1, q2), max(q1, q2))
        interaction_weight[key] += 1.0
        logical_degree[q1] += 1.0
        logical_degree[q2] += 1.0
    
    # Build logical adjacency
    logical_adj = defaultdict(set)
    for (q1, q2) in interaction_weight:
        logical_adj[q1].add(q2)
        logical_adj[q2].add(q1)
    
    # --- Step 1: Decompose hardware graph into hexagonal cells ---
    # In heavy-hex: degree-3 nodes are backbone, degree-2 nodes are chain/bridge
    # We find connected components of degree-3 nodes and their degree-2 neighbors
    
    degree = {pq: len(self.backend[pq]) for pq in physical_qubits}
    
    # Find cells by BFS from degree-3 nodes
    # A cell = a group of nearby physical qubits forming a local cluster
    # Use a simple greedy clustering: seed from degree-3 nodes, grow cells
    
    backbone_nodes = [pq for pq in physical_qubits if degree.get(pq, 0) >= 3]
    chain_nodes = [pq for pq in physical_qubits if degree.get(pq, 0) <= 2]
    
    # Build cells via connected component clustering on backbone subgraph
    # then attach chain nodes to nearest backbone cell
    
    # First find connected components of backbone nodes (within distance 2 of each other)
    backbone_set = set(backbone_nodes)
    backbone_adj = defaultdict(set)
    for bn in backbone_nodes:
        for nb in self.backend[bn]:
            if nb in backbone_set:
                backbone_adj[bn].add(nb)
            else:
                # Check if nb connects to another backbone node (bridge)
                for nb2 in self.backend[nb]:
                    if nb2 in backbone_set and nb2 != bn:
                        backbone_adj[bn].add(nb2)
    
    # Find connected components of backbone graph
    visited_backbone = set()
    backbone_components = []
    for bn in backbone_nodes:
        if bn in visited_backbone:
            continue
        comp = set()
        queue = deque([bn])
        while queue:
            node = queue.popleft()
            if node in comp:
                continue
            comp.add(node)
            visited_backbone.add(node)
            for nb in backbone_adj[node]:
                if nb not in comp:
                    queue.append(nb)
        backbone_components.append(comp)
    
    # If backbone detection didn't work well (no degree-3 nodes), fall back to 
    # generic graph partitioning
    if len(backbone_components) == 0:
        # Fallback: partition physical qubits into roughly equal cells
        target_cell_size = max(8, min(16, n_logical // 3 + 1))
        cells = []
        remaining = set(physical_qubits)
        while remaining:
            seed = min(remaining)  # deterministic seed
            cell = set()
            queue = deque([seed])
            while queue and len(cell) < target_cell_size:
                node = queue.popleft()
                if node not in remaining:
                    continue
                cell.add(node)
                remaining.discard(node)
                for nb in sorted(self.backend.get(node, set())):
                    if nb in remaining and nb not in cell:
                        queue.append(nb)
            cells.append(cell)
    else:
        # Build cells: each backbone component + its chain neighbors
        cells = []
        assigned_physical = set()
        
        # Sort components by size for determinism
        backbone_components.sort(key=lambda c: (len(c), min(c)))
        
        for comp in backbone_components:
            cell = set(comp)
            assigned_physical.update(comp)
            cells.append(cell)
        
        # Assign chain nodes to nearest cell (by distance)
        for cn in chain_nodes:
            if cn in assigned_physical:
                continue
            best_cell = 0
            best_dist = float('inf')
            for ci, cell in enumerate(cells):
                for pq in cell:
                    d = self.distance_matrix[cn][pq]
                    if d < best_dist:
                        best_dist = d
                        best_cell = ci
            cells[best_cell].add(cn)
            assigned_physical.add(cn)
        
        # Assign any remaining physical qubits
        for pq in physical_qubits:
            if pq not in assigned_physical:
                best_cell = 0
                best_dist = float('inf')
                for ci, cell in enumerate(cells):
                    for cpq in cell:
                        d = self.distance_matrix[pq][cpq]
                        if d < best_dist:
                            best_dist = d
                            best_cell = ci
                cells[best_cell].add(pq)
    
    n_cells = len(cells)
    
    # If only one cell or trivial case, do simple placement
    if n_cells <= 1 or n_logical <= 1:
        mapping = list(range(num_q))
        reverse_mapping = list(range(num_q))
        self.mapping_dict = mapping
        self.reverse_mapping_dict = reverse_mapping
        if self.use_isl:
            from src.utils.isl_data_loader import dict_to_isl_map
            self.isl_mapping = dict_to_isl_map(self.mapping_dict)
        return
    
    # --- Step 2: Partition logical qubits into clusters matching cell sizes ---
    # Use greedy weighted graph partitioning (Kernighan-Lin-style)
    
    # Target: n_cells clusters, sizes proportional to cell sizes
    cell_sizes = [len(c) for c in cells]
    total_physical = sum(cell_sizes)
    
    # Determine target cluster sizes proportional to cell sizes
    # but total logical qubits may be less than total physical
    target_sizes = []
    remaining_logical = n_logical
    for i in range(n_cells):
        if i == n_cells - 1:
            target_sizes.append(remaining_logical)
        else:
            sz = max(0, round(n_logical * cell_sizes[i] / total_physical))
            sz = min(sz, remaining_logical)
            target_sizes.append(sz)
            remaining_logical -= sz
    
    # Greedy BFS partitioning of logical qubits based on interaction graph
    clusters = []
    assigned_logical = set()
    
    # Sort logical qubits by interaction degree (descending)
    lq_by_degree = sorted(logical_qubits, key=lambda q: -logical_degree.get(q, 0))
    
    for ci in range(n_cells):
        sz = target_sizes[ci]
        if sz <= 0:
            clusters.append(set())
            continue
        
        # Find best seed: unassigned logical qubit with highest connectivity 
        # to already-assigned qubits in this cluster (or highest degree if first)
        seed = None
        for lq in lq_by_degree:
            if lq not in assigned_logical:
                seed = lq
                break
        
        if seed is None:
            clusters.append(set())
            continue
        
        cluster = set()
        cluster.add(seed)
        assigned_logical.add(seed)
        
        # BFS grow weighted by interaction
        while len(cluster) < sz:
            best_q = None
            best_score = -1
            for lq in logical_qubits:
                if lq in assigned_logical:
                    continue
                score = 0
                for cq in cluster:
                    key = (min(lq, cq), max(lq, cq))
                    score += interaction_weight.get(key, 0)
                if score > best_score:
                    best_score = score
                    best_q = lq
            if best_q is None:
                break
            cluster.add(best_q)
            assigned_logical.add(best_q)
        
        clusters.append(cluster)
    
    # Assign any remaining logical qubits
    for lq in logical_qubits:
        if lq not in assigned_logical:
            # Find cluster with most interaction and space
            best_ci = 0
            best_score = -1
            for ci, cl in enumerate(clusters):
                score = sum(interaction_weight.get((min(lq, cq), max(lq, cq)), 0) for cq in cl)
                if score > best_score:
                    best_score = score
                    best_ci = ci
            clusters[best_ci].add(lq)
            assigned_logical.add(lq)
    
    # --- Step 3: Cell-to-Cluster assignment via Hungarian matching ---
    # Cost[ci][cj] = sum of inter-cluster interaction * inter-cell distance
    # Simplified: cost = negative of (cluster interaction density * cell proximity)
    
    # Compute cluster centroids in interaction space and cell centroids in physical space
    # Use a simpler cost: for each (cluster_i, cell_j), compute the sum of
    # interaction weights within cluster_i * average intra-cell distance of cell_j
    # Plus penalty for size mismatch
    
    # Compute cell centroid physical qubit (median)
    cell_centroids = []
    for cell in cells:
        cell_list = sorted(cell)
        if not cell_list:
            cell_centroids.append(0)
            continue
        # Pick the qubit that minimizes sum of distances to all others in cell
        best_pq = cell_list[0]
        best_dist = float('inf')
        for pq in cell_list:
            d = sum(self.distance_matrix[pq][pq2] for pq2 in cell_list)
            if d < best_dist:
                best_dist = d
                best_pq = pq
        cell_centroids.append(best_pq)
    
    # Compute inter-cell distance matrix (centroid to centroid)
    inter_cell_dist = [[0.0]*n_cells for _ in range(n_cells)]
    for i in range(n_cells):
        for j in range(n_cells):
            inter_cell_dist[i][j] = self.distance_matrix[cell_centroids[i]][cell_centroids[j]]
    
    # Compute inter-cluster interaction
    inter_cluster_interaction = [[0.0]*n_cells for _ in range(n_cells)]
    # Build cluster index lookup
    lq_to_cluster = {}
    for ci, cl in enumerate(clusters):
        for lq in cl:
            lq_to_cluster[lq] = ci
    
    for (q1, q2), w in interaction_weight.items():
        c1 = lq_to_cluster.get(q1, -1)
        c2 = lq_to_cluster.get(q2, -1)
        if c1 != c2 and c1 >= 0 and c2 >= 0:
            inter_cluster_interaction[c1][c2] += w
            inter_cluster_interaction[c2][c1] += w
    
    # Hungarian cost matrix: assign cluster i to cell j
    # Cost = sum over other clusters k: inter_cluster_interaction[i][k] * inter_cell_dist[j][assigned_cell_of_k]
    # Since we don't know assignments yet, approximate:
    # Cost(i,j) = sum_k inter_cluster_interaction[i][k] * min_cell_dist_from_j
    # Simpler: use sum of inter_cluster interaction * cell centroid distance
    
    # Build cost matrix for assignment
    cost_matrix = [[0.0]*n_cells for _ in range(n_cells)]
    for ci in range(n_cells):
        cluster_interaction_total = sum(
            interaction_weight.get((min(q1, q2), max(q1, q2)), 0)
            for q1 in clusters[ci] for q2 in clusters[ci] if q1 < q2
        )
        for cj in range(n_cells):
            # Intra-cell cost: interaction * avg distance within cell
            cell_list = sorted(cells[cj])
            if len(cell_list) <= 1:
                avg_cell_dist = 0
            else:
                total_d = sum(self.distance_matrix[p1][p2] for p1 in cell_list for p2 in cell_list if p1 < p2)
                avg_cell_dist = total_d / (len(cell_list) * (len(cell_list)-1) / 2)
            
            # Size mismatch penalty
            size_penalty = abs(len(clusters[ci]) - len(cells[cj])) * 10.0
            
            cost_matrix[ci][cj] = cluster_interaction_total * avg_cell_dist + size_penalty
    
    # Simple Hungarian-like assignment (greedy for small n_cells)
    # For small n (typically 5-15), greedy with refinement is fine
    assignment = list(range(n_cells))  # cluster i -> cell i initially
    
    if n_cells <= 10:
        # Try all permutations for very small n, or use greedy
        # Greedy: assign cluster with highest interaction first to best cell
        cluster_total_interaction = []
        for ci in range(n_cells):
            total = sum(inter_cluster_interaction[ci])
            total += sum(
                interaction_weight.get((min(q1, q2), max(q1, q2)), 0)
                for q1 in clusters[ci] for q2 in clusters[ci] if q1 < q2
            )
            cluster_total_interaction.append((total, ci))
        cluster_total_interaction.sort(reverse=True)
        
        used_cells = set()
        assignment = [0] * n_cells
        for _, ci in cluster_total_interaction:
            best_cj = 0
            best_cost = float('inf')
            for cj in range(n_cells):
                if cj in used_cells:
                    continue
                if cost_matrix[ci][cj] < best_cost:
                    best_cost = cost_matrix[ci][cj]
                    best_cj = cj
            assignment[ci] = best_cj
            used_cells.add(best_cj)
    else:
        # For larger n, use scipy-like Hungarian or greedy
        # Greedy assignment
        used_cells = set()
        assignment = [0] * n_cells
        order = sorted(range(n_cells), key=lambda ci: len(clusters[ci]), reverse=True)
        for ci in order:
            best_cj = 0
            best_cost = float('inf')
            for cj in range(n_cells):
                if cj in used_cells:
                    continue
                if cost_matrix[ci][cj] < best_cost:
                    best_cost = cost_matrix[ci][cj]
                    best_cj = cj
            assignment[ci] = best_cj
            used_cells.add(best_cj)
    
    # --- Step 4: Intra-cell QAP for each (cluster, cell) pair ---
    mapping = list(range(num_q))
    reverse_mapping = list(range(num_q))
    
    for ci in range(n_cells):
        cell_idx = assignment[ci]
        cluster_lqs = sorted(clusters[ci])
        cell_pqs = sorted(cells[cell_idx])
        
        if len(cluster_lqs) == 0:
            continue
        
        n_local = len(cluster_lqs)
        n_cell = len(cell_pqs)
        
        # Build local interaction weights
        local_interactions = []
        for i in range(n_local):
            for j in range(i+1, n_local):
                q1, q2 = cluster_lqs[i], cluster_lqs[j]
                key = (min(q1, q2), max(q1, q2))
                w = interaction_weight.get(key, 0)
                if w > 0:
                    local_interactions.append((i, j, w))
        
        if n_local <= n_cell:
            # Need to pick n_local physical qubits from cell and assign
            # For small sizes, use greedy + local search
            
            # Sort cell physical qubits by centrality within cell
            cell_centrality = {}
            for pq in cell_pqs:
                cell_centrality[pq] = sum(self.distance_matrix[pq][pq2] for pq2 in cell_pqs)
            cell_pqs_sorted = sorted(cell_pqs, key=lambda pq: cell_centrality[pq])
            
            # Sort logical qubits by degree
            cluster_lqs_sorted = sorted(cluster_lqs, key=lambda q: -logical_degree.get(q, 0))
            
            # Initial greedy assignment: highest degree logical -> most central physical
            selected_pqs = cell_pqs_sorted[:n_local]
            local_mapping = {}  # local index -> physical qubit
            for i, lq in enumerate(cluster_lqs_sorted):
                local_mapping[lq] = selected_pqs[i]
            
            # Compute QAP cost
            def qap_cost(lmap):
                cost = 0
                for i, j, w in local_interactions:
                    q1, q2 = cluster_lqs[i], cluster_lqs[j]
                    p1, p2 = lmap[q1], lmap[q2]
                    cost += w * self.distance_matrix[p1][p2]
                return cost
            
            best_cost = qap_cost(local_mapping)
            best_mapping = dict(local_mapping)
            
            # 2-opt local search refinement
            improved = True
            max_iters = 50
            iters = 0
            while improved and iters < max_iters:
                improved = False
                iters += 1
                for i in range(n_local):
                    for j in range(i+1, n_local):
                        lq_i = cluster_lqs_sorted[i]
                        lq_j = cluster_lqs_sorted[j]
                        # Try swap
                        local_mapping[lq_i], local_mapping[lq_j] = local_mapping[lq_j], local_mapping[lq_i]
                        new_cost = qap_cost(local_mapping)
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_mapping = dict(local_mapping)
                            improved = True
                        else:
                            local_mapping[lq_i], local_mapping[lq_j] = local_mapping[lq_j], local_mapping[lq_i]
            
            # Also try swapping with unused physical qubits in the cell
            unused_pqs = [pq for pq in cell_pqs if pq not in best_mapping.values()]
            local_mapping = dict(best_mapping)
            for upq in unused_pqs:
                for lq in cluster_lqs:
                    old_pq = local_mapping[lq]
                    local_mapping[lq] = upq
                    new_cost = qap_cost(local_mapping)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_mapping = dict(local_mapping)
                    else:
                        local_mapping[lq] = old_pq
            
            # Apply best mapping
            for lq, pq in best_mapping.items():
                mapping[lq] = pq
                reverse_mapping[pq] = lq
    
    # Handle unmapped logical qubits (assign to remaining physical qubits)
    mapped_physical = set()
    mapped_logical = set()
    for lq in logical_qubits:
        if mapping[lq] != lq or lq in [m_lq for m_lq in logical_qubits if mapping[m_lq] != m_lq]:
            mapped_physical.add(mapping[lq])
            mapped_logical.add(lq)
    
    # Re-check: explicitly track what was assigned by the QAP
    assigned_lqs = set()
    assigned_pqs = set()
    for ci in range(n_cells):
        cell_idx = assignment[ci]
        cluster_lqs = sorted(clusters[ci])
        for lq in cluster_lqs:
            if lq in logical_qubit_set:
                assigned_lqs.add(lq)
                assigned_pqs.add(mapping[lq])
    
    # Assign unassigned logical qubits to unassigned physical qubits
    unassigned_lqs = [lq for lq in logical_qubits if lq not in assigned_lqs]
    unassigned_pqs = [pq for pq in physical_qubits if pq not in assigned_pqs]
    
    for lq in unassigned_lqs:
        if unassigned_pqs:
            pq = unassigned_pqs.pop(0)
            mapping[lq] = pq
            reverse_mapping[pq] = lq
            assigned_lqs.add(lq)
            assigned_pqs.add(pq)
    
    # --- Step 5: Cross-cell 2-opt refinement on boundary qubits ---
    # Find boundary qubits: logical qubits with interactions to other clusters
    boundary_lqs = set()
    for (q1, q2), w in interaction_weight.items():
        c1 = lq_to_cluster.get(q1, -1)
        c2 = lq_to_cluster.get(q2, -1)
        if c1 != c2:
            boundary_lqs.add(q1)
            boundary_lqs.add(q2)
    
    boundary_lqs = sorted(boundary_lqs)
    
    # 2-opt on boundary qubits
    improved = True
    max_refinement_iters = 30
    ref_iter = 0
    
    def total_cost(m):
        cost = 0
        for (q1, q2), w in interaction_weight.items():
            cost += w * self.distance_matrix[m[q1]][m[q2]]
        return cost
    
    current_cost = total_cost(mapping)
    
    while improved and ref_iter < max_refinement_iters:
        improved = False
        ref_iter += 1
        for i in range(len(boundary_lqs)):
            for j in range(i+1, len(boundary_lqs)):
                lq_i = boundary_lqs[i]
                lq_j = boundary_lqs[j]
                pq_i = mapping[lq_i]
                pq_j = mapping[lq_j]
                
                # Compute delta cost of swap
                delta = 0
                for lq_k in logical_qubits:
                    if lq_k == lq_i or lq_k == lq_j:
                        continue
                    pq_k = mapping[lq_k]
                    
                    key_ik = (min(lq_i, lq_k), max(lq_i, lq_k))
                    w_ik = interaction_weight.get(key_ik, 0)
                    if w_ik > 0:
                        delta += w_ik * (self.distance_matrix[pq_j][pq_k] - self.distance_matrix[pq_i][pq_k])
                    
                    key_jk = (min(lq_j, lq_k), max(lq_j, lq_k))
                    w_jk = interaction_weight.get(key_jk, 0)
                    if w_jk > 0:
                        delta += w_jk * (self.distance_matrix[pq_i][pq_k] - self.distance_matrix[pq_j][pq_k])
                
                # Check i-j interaction (distance doesn't change for the pair itself)
                
                if delta < -1e-9:
                    mapping[lq_i] = pq_j
                    mapping[lq_j] = pq_i
                    reverse_mapping[pq_i] = lq_j
                    reverse_mapping[pq_j] = lq_i
                    current_cost += delta
                    improved = True
    
    # --- Final: ensure valid bijection for all qubits ---
    # The mapping/reverse_mapping was initialized as identity.
    # We modified entries for logical qubits. Now fix collisions.
    
    # Rebuild clean mapping
    final_mapping = [-1] * num_q
    final_reverse = [-1] * num_q
    
    used_pqs = set()
    # First assign all logical qubits that were in the circuit
    for lq in sorted(logical_qubit_set):
        pq = mapping[lq]
        if pq not in used_pqs:
            final_mapping[lq] = pq
            final_reverse[pq] = lq
            used_pqs.add(pq)
        else:
            # Collision: find nearest unused physical qubit
            best_pq = None
            best_d = float('inf')
            for p in physical_qubits:
                if p not in used_pqs:
                    d = self.distance_matrix[pq][p] if pq < len(self.distance_matrix) and p < len(self.distance_matrix[0]) else float('inf')
                    if d < best_d:
                        best_d = d
                        best_pq = p
            if best_pq is not None:
                final_mapping[lq] = best_pq
                final_reverse[best_pq] = lq
                used_pqs.add(best_pq)
    
    # Assign remaining logical qubits (not in circuit) to remaining physical qubits
    remaining_pqs = sorted(set(physical_qubits) - used_pqs)
    remaining_idx = 0
    for lq in range(num_q):
        if final_mapping[lq] == -1:
            if remaining_idx < len(remaining_pqs):
                pq = remaining_pqs[remaining_idx]
                final_mapping[lq] = pq
                final_reverse[pq] = lq
                remaining_idx += 1
    
    # Handle edge case: if num_q > len(physical_qubits), some logical qubits 
    # might still be -1. Assign them to any remaining slots.
    all_pqs = set(range(num_q))
    used_final = set(pq for pq in final_mapping if pq >= 0)
    remaining_all = sorted(all_pqs - used_final)
    ri = 0
    for lq in range(num_q):
        if final_mapping[lq] == -1:
            if ri < len(remaining_all):
                pq = remaining_all[ri]
                final_mapping[lq] = pq
                final_reverse[pq] = lq
                ri += 1
    
    self.mapping_dict = final_mapping
    self.reverse_mapping_dict = final_reverse
    
    if self.use_isl:
        from src.utils.isl_data_loader import dict_to_isl_map
        self.isl_mapping = dict_to_isl_map(self.mapping_dict)