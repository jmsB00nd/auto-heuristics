def init_mapping(self):
    import random

    N = self.num_qubits
    dist = self.distance_matrix

    interactions = {}
    for _gid, qubits in self.access.items():
        if len(qubits) == 2:
            a, b = qubits[0], qubits[1]
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            interactions[key] = interactions.get(key, 0) + 1

    interactions = {(a, b): w for (a, b), w in interactions.items()
                    if 0 <= a < N and 0 <= b < N}

    def fitness(perm):
        s = 0.0
        for (a, b), w in interactions.items():
            s += w * dist[perm[a]][perm[b]]
        return s

    def random_perm():
        p = list(range(N))
        random.shuffle(p)
        return p

    def tournament(pop, fits, k=3):
        if len(pop) <= 1:
            return list(pop[0]) if pop else random_perm()
        kk = min(k, len(pop))
        idxs = random.sample(range(len(pop)), kk)
        best_i = min(idxs, key=lambda i: fits[i])
        return list(pop[best_i])

    def pmx(p1, p2):
        size = len(p1)
        if size < 2:
            return list(p1)
        child = [-1] * size
        a, b = sorted(random.sample(range(size), 2))
        for i in range(a, b + 1):
            child[i] = p1[i]
        seg_set = set(child[a:b + 1])
        pos_p2 = {v: i for i, v in enumerate(p2)}
        for i in range(a, b + 1):
            val = p2[i]
            if val in seg_set:
                continue
            j = i
            guard = 0
            while a <= j <= b and guard < size + 1:
                j = pos_p2[p1[j]]
                guard += 1
            if 0 <= j < size and child[j] == -1:
                child[j] = val
        for i in range(size):
            if child[i] == -1:
                child[i] = p2[i]
        used = set()
        dup = False
        for v in child:
            if v in used:
                dup = True
                break
            used.add(v)
        if dup or len(set(child)) != size:
            return list(p1)
        return child

    def cycle_crossover(p1, p2):
        size = len(p1)
        if size < 2:
            return list(p1)
        child = [-1] * size
        idx_p2 = {v: i for i, v in enumerate(p2)}
        i = 0
        guard = 0
        while child[i] == -1 and guard < size + 1:
            child[i] = p1[i]
            nxt = idx_p2.get(p1[i], i)
            if nxt == i:
                break
            i = nxt
            guard += 1
        for j in range(size):
            if child[j] == -1:
                child[j] = p2[j]
        if len(set(child)) != size:
            return list(p1)
        return child

    def mutate(perm, rate=0.2):
        if len(perm) >= 2 and random.random() < rate:
            i, j = random.sample(range(len(perm)), 2)
            perm[i], perm[j] = perm[j], perm[i]
        return perm

    pop_size = 30 if N >= 4 else max(4, N)
    generations = 40 if N >= 4 else 10

    population = []
    try:
        from src.mapping.initial_mapping import generate_structure_aware_initial_mapping
        seed_map, _ = generate_structure_aware_initial_mapping(
            self.access, self.backend, self.distance_matrix, self.num_qubits
        )
        seed_list = list(seed_map)
        if len(seed_list) == N and len(set(seed_list)) == N and all(0 <= v < N for v in seed_list):
            population.append(seed_list)
    except Exception:
        pass

    population.append(list(range(N)))
    while len(population) < pop_size:
        population.append(random_perm())

    fits = [fitness(p) for p in population]
    best_i = min(range(len(population)), key=lambda i: fits[i])
    best = list(population[best_i])
    best_fit = fits[best_i]

    for _gen in range(generations):
        new_pop = [list(best)]
        while len(new_pop) < pop_size:
            p1 = tournament(population, fits)
            p2 = tournament(population, fits)
            if random.random() < 0.5:
                child = pmx(p1, p2)
            else:
                child = cycle_crossover(p1, p2)
            child = mutate(child, rate=0.25)
            if len(child) == N and len(set(child)) == N:
                new_pop.append(child)
            else:
                new_pop.append(random_perm())
        population = new_pop
        fits = [fitness(p) for p in population]
        cur_i = min(range(len(population)), key=lambda i: fits[i])
        if fits[cur_i] < best_fit:
            best_fit = fits[cur_i]
            best = list(population[cur_i])

    if len(best) != N or len(set(best)) != N or any(not (0 <= v < N) for v in best):
        best = list(range(N))

    self.mapping_dict = list(best)
    self.reverse_mapping_dict = [0] * N
    for logical_q, physical_q in enumerate(self.mapping_dict):
        self.reverse_mapping_dict[physical_q] = logical_q

    assert len(set(self.mapping_dict)) == len(self.mapping_dict)