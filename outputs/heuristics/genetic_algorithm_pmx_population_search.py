def init_mapping(self):
    import random
    from src.mapping.initial_mapping import generate_structure_aware_initial_mapping

    N = self.num_qubits
    dist = self.distance_matrix

    interactions = {}
    active = set()
    for gate_id, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            interactions[key] = interactions.get(key, 0) + 1
            active.add(a); active.add(b)
    inter_list = [(a, b, w) for (a, b), w in interactions.items()]

    def fitness(perm):
        total = 0.0
        for a, b, w in inter_list:
            if a < N and b < N:
                pa = perm[a]; pb = perm[b]
                total += w * dist[pa][pb]
        return total

    def random_perm():
        p = list(range(N))
        random.shuffle(p)
        return p

    def pmx(p1, p2):
        if N < 2:
            return list(p1)
        i, j = sorted(random.sample(range(N), 2))
        child = [-1] * N
        for k in range(i, j + 1):
            child[k] = p1[k]
        seg_set = set(child[i:j + 1])
        pos_in_p1 = {v: idx for idx, v in enumerate(p1)}
        for k in range(i, j + 1):
            v = p2[k]
            if v in seg_set:
                continue
            pos = k
            while True:
                conflict = p1[pos]
                pos = p2.index(conflict)
                if pos < i or pos > j:
                    child[pos] = v
                    break
        for k in range(N):
            if child[k] == -1:
                child[k] = p2[k]
        return child

    def mutate(perm):
        if N < 2:
            return perm
        if random.random() < 0.5 and inter_list:
            a, b, _ = random.choice(inter_list)
            if a < N and b < N:
                pa = perm[a]
                neighbors = list(self.backend.get(pa, set()))
                if neighbors:
                    target_phys = random.choice(neighbors)
                    if target_phys != perm[b]:
                        ia, ib = perm.index(pa), perm.index(target_phys)
                        perm[ia], perm[ib] = perm[ib], perm[ia]
                        return perm
        i, j = random.sample(range(N), 2)
        perm[i], perm[j] = perm[j], perm[i]
        return perm

    def tournament(pop, fits, k=3):
        idxs = random.sample(range(len(pop)), min(k, len(pop)))
        best = min(idxs, key=lambda x: fits[x])
        return pop[best]

    pop_size = max(8, min(24, N))
    generations = 30
    population = [random_perm() for _ in range(pop_size)]

    try:
        warm_md, _ = generate_structure_aware_initial_mapping(
            self.access, self.backend, self.distance_matrix, self.num_qubits
        )
        warm = list(warm_md)
        seen = set(); ok = True
        for v in warm:
            if v in seen or not (0 <= v < N):
                ok = False; break
            seen.add(v)
        if ok and len(warm) == N:
            population[0] = warm
    except Exception:
        pass

    fits = [fitness(p) for p in population]

    for _ in range(generations):
        children = []
        num_children = max(2, pop_size // 4)
        for _c in range(num_children):
            p1 = tournament(population, fits)
            p2 = tournament(population, fits)
            child = pmx(p1, p2)
            if random.random() < 0.3:
                child = mutate(list(child))
            children.append(child)
        child_fits = [fitness(c) for c in children]
        order = sorted(range(len(population)), key=lambda i: fits[i], reverse=True)
        for c, cf in zip(children, child_fits):
            worst_idx = order.pop(0)
            if cf < fits[worst_idx]:
                population[worst_idx] = c
                fits[worst_idx] = cf

    best_idx = min(range(len(population)), key=lambda i: fits[i])
    best = population[best_idx]

    self.mapping_dict = list(best)
    self.reverse_mapping_dict = [0] * N
    for logical, physical in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[physical] = logical

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)