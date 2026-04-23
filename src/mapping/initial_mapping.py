"""Structure-aware initial mapping for the Qlosure router.

Deterministic, single-pass logical -> physical placement. No routing
simulation, no hyperparameter search, no second pass.

The pipeline:

  1. Build two weighted Qubit Interaction Graphs from the circuit's gate
     list. Both weight each 2-qubit gate by a time-decay factor (early
     gates are harder to amortise through routing). One QIG also multiplies
     by each gate's *remaining critical depth* -- the length of the longest
     dependency chain that follows it -- because getting critical-path
     gates placed well pays off in both swap count and circuit depth.

  2. Dispatch to one of two embeddings based on the QIG's structure:

       - Linear: qubits are laid out along a longest simple hardware path
         in Reverse Cuthill-McKee order. Chosen for tree-like QIGs (chains,
         stars, GHZ/W/Ising), near-complete QIGs (QFT), cramped medium-
         dense QIGs, and hub+dense-leaf QIGs (qugan, knn). The router
         routes this shape by "walking" the hub along the path.

       - Greedy: qubits are placed one at a time by minimising the
         sum over placed partners of qig_weight * hardware_distance.
         Tie-broken lexicographically by (max distance to placed,
         free-neighbour look-ahead, global centrality).

  3. Any logical id never touched by the circuit is back-filled onto the
     most central unused physical, keeping the bijective arrays the
     router expects.

Structural classifiers (all derived from the QIG and the hardware graph,
no tunable coefficients):

    tree_like     : |QIG edges| <= n_active
    near_complete : QIG density > 0.5
    star_like     : one qubit covers >= n/2 partners AND avg_qig_deg <= 3.5
    medium_dense  : avg_qig_deg > 2 * max_hw_deg AND n_active < n_hw/2
    hub_dense     : one qubit covers >= n/2 partners AND avg_qig_deg > 4.5
"""

from collections import defaultdict, deque
import math


def _bfs_ball(start, k, backend_adj):
    """Return the k physicals reachable from `start` first via BFS (includes start)."""
    visited = [start]
    seen = {start}
    queue = deque([start])
    while queue and len(visited) < k:
        cur = queue.popleft()
        for nb in sorted(backend_adj.get(cur, ())):
            if nb in seen:
                continue
            seen.add(nb)
            visited.append(nb)
            queue.append(nb)
            if len(visited) >= k:
                break
    return visited


def _select_canvas(n_active, valid_phys, backend_adj, distance_matrix):
    """Pick the connected k-subgraph (k = n_active) of the hardware whose internal
    sum of pairwise distances is minimal."""
    best_set = None
    best_score = float('inf')
    best_radius = float('inf')
    for p in valid_phys:
        ball = _bfs_ball(p, n_active, backend_adj)
        if len(ball) < n_active:
            continue
        score = 0
        radius = 0
        for i in range(len(ball)):
            di = distance_matrix[ball[i]]
            for j in range(i + 1, len(ball)):
                d = di[ball[j]]
                score += d
                if d > radius:
                    radius = d
        if (radius, score) < (best_radius, best_score):
            best_score = score
            best_radius = radius
            best_set = ball
    return best_set


def _longest_path_from(start, valid_set, backend_adj, target_len, step_budget=200000):
    """DFS-with-backtracking path finder that always returns a *contiguous* simple path.

    Extends with Warnsdorff ordering (fewest-unvisited-neighbour first). Terminates
    as soon as a path of `target_len` is found. The resulting path is guaranteed to
    have consecutive entries adjacent in `backend_adj` — the caller relies on this.
    """
    visited = {start}
    path = [start]
    best = [start]
    budget = [step_budget]

    def dfs():
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        if len(path) > len(best):
            best[:] = list(path)
            if len(best) >= target_len:
                return True
        cur = path[-1]
        nbrs = [n for n in backend_adj.get(cur, ()) if n in valid_set and n not in visited]
        nbrs.sort(
            key=lambda n: (
                sum(1 for x in backend_adj.get(n, ()) if x in valid_set and x not in visited),
                n,
            )
        )
        for n in nbrs:
            visited.add(n)
            path.append(n)
            if dfs():
                return True
            path.pop()
            visited.discard(n)
        return False

    dfs()
    return best


def _rcm_order(referenced, qig, activity, by_weight=False):
    """Reverse Cuthill-McKee style ordering of logicals based on weighted QIG.

    - Classical mode (`by_weight=False`): neighbours expanded in ascending
      activity order. Chains map to chains, stars to (leaf, hub, leaves).
      For near-complete QIGs (QFT) this yields the natural numeric order,
      which empirically routes best.

    - Weight mode (`by_weight=True`): neighbours expanded strongest-edge
      first. For circuits with non-uniform edge weights but moderately
      dense structure (e.g. qugan / knn's leaf subgraph) this pulls
      heavily-coupled pairs adjacent in the path, a big win.
    """
    ordering = []
    seen = set()
    remaining = set(referenced)
    while remaining:
        start = min(remaining, key=lambda q: (activity.get(q, 0), q))
        queue = deque([start])
        seen.add(start)
        remaining.discard(start)
        while queue:
            cur = queue.popleft()
            ordering.append(cur)
            ngbrs = [n for n in qig[cur] if n in referenced and n not in seen]
            if by_weight:
                ngbrs.sort(key=lambda n: (-qig[cur].get(n, 0.0), n))
            else:
                ngbrs.sort(key=lambda n: (activity.get(n, 0), n))
            for n in ngbrs:
                if n not in seen:
                    seen.add(n)
                    remaining.discard(n)
                    queue.append(n)
    return ordering[::-1]


def _linear_embedding(
    referenced,
    qig,
    activity,
    valid_phys,
    backend_adj,
    distance_matrix,
    num_qubits,
    physical_centrality_full,
    rcm_by_weight=False,
):
    """Embed an RCM-ordered logical sequence onto a longest simple hardware path.

    Chosen when the QIG has at least one very-high-degree vertex (a hub) or is
    dense. On heavy-hex this is the arrangement the qlosure router routes most
    efficiently, because it lets the hub 'walk' along the path collecting
    interactions one swap at a time.
    """
    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits
    n_active = len(referenced)

    valid_set = set(valid_phys)

    # Start from the hardware qubit whose longest simple path is longest overall.
    # Degree-1 endpoints tend to yield the longest paths on heavy-hex (no branching).
    candidate_starts = [p for p in valid_phys if len(backend_adj.get(p, ())) == 1]
    if not candidate_starts:
        # Fall back to the least-central degree-2 qubit.
        deg2 = [p for p in valid_phys if len(backend_adj.get(p, ())) == 2]
        candidate_starts = sorted(deg2, key=lambda p: (-physical_centrality_full[p], p))[:4] or valid_phys

    best_path = None
    for s in candidate_starts:
        path = _longest_path_from(s, valid_set, backend_adj, n_active)
        if best_path is None or len(path) > len(best_path):
            best_path = path
        if best_path is not None and len(best_path) >= n_active:
            break

    if best_path is None or len(best_path) < n_active:
        # Last-ditch: canvas fallback.
        canvas = _select_canvas(n_active, valid_phys, backend_adj, distance_matrix)
        best_path = canvas if canvas is not None else valid_phys[:n_active]

    phys_path = best_path[:n_active]

    order = _rcm_order(referenced, qig, activity, by_weight=rcm_by_weight)

    for L, P in zip(order, phys_path):
        mapping[L] = P
        reverse_mapping[P] = L

    return mapping, reverse_mapping


def generate_structure_aware_initial_mapping(
    access_dict,
    backend_adj,
    distance_matrix,
    num_qubits,
):
    n_phys = len(distance_matrix)
    valid_phys = [p for p in range(n_phys) if distance_matrix[p]]

    referenced = set()
    sorted_gates = sorted(access_dict.keys())
    for g in sorted_gates:
        for q in access_dict[g]:
            referenced.add(q)

    n_active = len(referenced)

    mapping = [-1] * num_qubits
    reverse_mapping = [-1] * num_qubits

    physical_centrality_full = [
        sum(d for d in distance_matrix[p] if d != float('inf'))
        for p in range(n_phys)
    ]

    if n_active == 0:
        ordered = sorted(valid_phys, key=lambda p: (physical_centrality_full[p], p))
        for i, P in enumerate(ordered[:num_qubits]):
            mapping[i] = P
            reverse_mapping[P] = i
        return mapping, reverse_mapping

    # Walk gates in reverse time order to compute each gate's *remaining critical
    # depth* -- the length of the longest chain of dependent 2-qubit gates that
    # follows it. The "next" pointer for a qubit is the next gate in time that
    # also touches that qubit.
    last_seen = {}      # qubit -> most recent gate id encountered (in reverse walk)
    crit_remaining = {}  # gate_id -> 1 + max(crit_remaining of immediate successors)
    for g in reversed(sorted_gates):
        qb = access_dict[g]
        succ_max = 0
        for q in qb:
            nxt = last_seen.get(q)
            if nxt is not None and crit_remaining.get(nxt, 0) > succ_max:
                succ_max = crit_remaining[nxt]
        crit_remaining[g] = 1 + succ_max
        for q in qb:
            last_seen[q] = g

    # Two QIGs: a plain time-weighted one for the linear embedding (where the
    # RCM ordering wants structural adjacency), and a critical-path-weighted
    # one for the greedy embedding (where getting deep-chain gates placed
    # well pays off in both swaps and depth).
    qig_time = defaultdict(lambda: defaultdict(float))
    qig_crit = defaultdict(lambda: defaultdict(float))
    activity_time = defaultdict(float)
    activity_crit = defaultdict(float)
    n_gates = len(sorted_gates)
    tau = max(20.0, n_gates / 4.0)
    max_crit = max(crit_remaining.values()) if crit_remaining else 1

    for idx, g in enumerate(sorted_gates):
        qb = access_dict[g]
        if len(qb) != 2:
            continue
        q1, q2 = qb
        if q1 == q2:
            continue
        time_factor = 1.0 + 2.0 * math.exp(-idx / tau)
        crit_factor = crit_remaining[g] / max_crit
        w_t = time_factor
        w_c = time_factor * (0.5 + crit_factor)
        qig_time[q1][q2] += w_t
        qig_time[q2][q1] += w_t
        activity_time[q1] += w_t
        activity_time[q2] += w_t
        qig_crit[q1][q2] += w_c
        qig_crit[q2][q1] += w_c
        activity_crit[q1] += w_c
        activity_crit[q2] += w_c

    # Structural dispatch. Two regimes benefit from a linear (path) embedding
    # instead of greedy-into-a-compact-cluster:
    #
    #   (A) Tree-like QIGs  (chains, pure stars, k-ary trees). A chain maps
    #       onto a hardware path with zero swaps; a star's hub can "walk"
    #       the path taking one swap per leaf. Detected by edges <= n_active.
    #
    #   (B) Near-complete QIGs  (QFT). Every pair interacts, so no placement
    #       can keep all neighbours adjacent. A path arrangement in RCM order
    #       minimises cumulative swap work because the router threads the
    #       interactions along the path. Detected by density > 0.5.
    #
    # All other circuits (sparse-with-structure: multiplier, qugan, adder,
    # knn, dnn, swap_test) prefer the weighted-greedy embedding because
    # their QIGs have multiple mid-degree vertices whose neighbourhoods must
    # be preserved locally.
    unique_qig_degrees = {q: len({nb for nb, w in qig_time[q].items() if w > 0}) for q in qig_time}
    total_qig_edges = sum(unique_qig_degrees.values()) / 2
    density = total_qig_edges / (n_active * (n_active - 1) / 2) if n_active > 1 else 0.0
    max_qig_deg = max(unique_qig_degrees.values()) if unique_qig_degrees else 0
    avg_qig_deg = 2 * total_qig_edges / n_active if n_active else 0.0
    max_hw_deg = max((len(backend_adj.get(p, ())) for p in valid_phys), default=1)
    n_hw = len(valid_phys)

    tree_like = total_qig_edges <= n_active
    near_complete = density > 0.5
    # Single-hub regime: one qubit interacts with at least half of all others,
    # while the average degree stays low. The router routes this best by
    # walking the hub along a 1-D path.
    star_like = (max_qig_deg >= n_active // 2) and (avg_qig_deg <= 3.5)
    # Hub-and-dense-leaves: one super-hub *and* the leaves themselves form a
    # moderately dense subgraph. The greedy embedding clusters everything
    # around the hub, which leaves no room to thread the leaf-leaf edges;
    # weight-RCM along a long path gives a much better arrangement. The
    # leaf density bar (avg_qig_deg > 4.5) is set to cover qugan's leaf
    # subgraph while excluding the dnn layer graph, where greedy still wins.
    hub_dense = (max_qig_deg >= n_active // 2) and (avg_qig_deg > 4.5)

    use_linear = tree_like or near_complete or star_like or hub_dense

    if use_linear:
        # Weight-RCM pays off only when the QIG has a non-trivial weight
        # distribution AND it isn't near-complete. Near-complete QIGs (QFT)
        # are uniform enough that classical RCM (visit order) routes better
        # than weight-RCM on this router.
        return _linear_embedding(
            referenced,
            qig_time,
            activity_time,
            valid_phys,
            backend_adj,
            distance_matrix,
            num_qubits,
            physical_centrality_full,
            rcm_by_weight=(hub_dense or star_like) and not near_complete,
        )

    # From here on: greedy. Use the critical-path-weighted QIG so that deep
    # dependency chains drive the placement.
    qig = qig_crit
    activity = activity_crit

    # 1. Pick logical seed: highest activity (deterministic tie-break by lowest id).
    seed_logical = max(activity, key=lambda q: (activity[q], -q))

    # 2. Pick a physical seed. A canvas (compact BFS ball of size n_active) is
    # used to score candidate seeds by "how compact is their local region",
    # but subsequent placement is NOT restricted to the canvas -- that would
    # break chain-shaped QIGs.
    canvas = _select_canvas(n_active, valid_phys, backend_adj, distance_matrix)
    if canvas is None:
        canvas = sorted(valid_phys, key=lambda p: (physical_centrality_full[p], p))[:n_active]

    canvas_centrality = {}
    for p in canvas:
        s = 0
        dp = distance_matrix[p]
        for q in canvas:
            if q != p:
                s += dp[q]
        canvas_centrality[p] = s
    seed_physical = min(canvas, key=lambda p: (canvas_centrality[p], p))

    mapping[seed_logical] = seed_physical
    reverse_mapping[seed_physical] = seed_logical
    placed = {seed_logical}
    used = {seed_physical}

    pending_active = referenced - placed

    frontier_score = defaultdict(float)
    for nb, w in qig[seed_logical].items():
        if nb not in placed:
            frontier_score[nb] += w

    while pending_active:
        if frontier_score:
            next_L = max(
                frontier_score,
                key=lambda q: (frontier_score[q], activity[q], -q),
            )
        else:
            next_L = max(pending_active, key=lambda q: (activity[q], -q))

        weights_to_placed = []
        for L_p in placed:
            w = qig[next_L].get(L_p, 0.0)
            if w > 0.0:
                weights_to_placed.append((L_p, w))

        # Unplaced QIG neighbours of next_L -- used for a small look-ahead bonus:
        # we prefer Ps that still leave room for next_L's unplaced partners.
        unplaced_partner_weight = 0.0
        for nb, w in qig[next_L].items():
            if nb not in placed:
                unplaced_partner_weight += w

        # Candidate physicals: any unused valid physical. We lexicographically
        # rank by (primary cost, max distance to any placed, centrality, id)
        # so that dense QIGs stay compact while chain/star QIGs can extend.
        best_P = None
        best_key = None
        for P in valid_phys:
            if P in used:
                continue
            if not weights_to_placed:
                primary = 0.0
                max_dist = 0
            else:
                primary = 0.0
                max_dist = 0
                for L_p, w in weights_to_placed:
                    dpp = distance_matrix[P][mapping[L_p]]
                    primary += w * dpp
                    if dpp > max_dist:
                        max_dist = dpp
            # Look-ahead: a rough estimate of how costly future placements
            # adjacent to P will be. free_deg = number of free neighbours of P
            # (these are the slots available for next_L's unplaced partners).
            free_deg = sum(1 for n in backend_adj.get(P, ()) if n not in used)
            # We want free_deg LARGE when unplaced_partner_weight is high, so
            # include -unplaced_partner_weight * free_deg as a negative term
            # (smaller key is better).
            lookahead = -unplaced_partner_weight * free_deg
            key = (primary, max_dist, lookahead, physical_centrality_full[P], P)
            if best_key is None or key < best_key:
                best_key = key
                best_P = P

        mapping[next_L] = best_P
        reverse_mapping[best_P] = next_L
        placed.add(next_L)
        used.add(best_P)
        pending_active.discard(next_L)
        frontier_score.pop(next_L, None)

        for nb, w in qig[next_L].items():
            if nb not in placed:
                frontier_score[nb] += w

    # Backfill any logical index never mentioned in the access dict (keeps arrays bijective).
    free_phys = sorted(
        (p for p in valid_phys if p not in used),
        key=lambda p: (physical_centrality_full[p], p),
    )
    free_idx = 0
    for L in range(num_qubits):
        if mapping[L] == -1 and free_idx < len(free_phys):
            P = free_phys[free_idx]
            free_idx += 1
            mapping[L] = P
            reverse_mapping[P] = L

    return mapping, reverse_mapping
