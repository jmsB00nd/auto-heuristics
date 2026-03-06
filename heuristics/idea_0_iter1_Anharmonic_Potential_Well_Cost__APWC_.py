# Idea: Anharmonic Potential Well Cost (APWC)
# Stats: {"mean_swaps": Infinity, "mean_depth": 0, "mean_runtime": 0, "total_circuits": 22, "successful_runs": 0, "failed_runs": 3, "error": "All circuits failed", "first_failure_error": "list index out of range", "first_failure_traceback": "Traceback (most recent call last):\n  File \"/home/jmsb00nd/Documents/auto-heuristics/orchestrator.py\", line 228, in inject_and_run\n    raise exception_container['error']\n  File \"/home/jmsb00nd/Documents/auto-heuristics/orchestrator.py\", line 210, in run_with_timeout\n    min_swaps, min_depth, _ = router.run(heuristic_method=\"Qlosure\")\n  File \"/home/jmsb00nd/Documents/auto-heuristics/src/mapping/routing.py\", line 102, in run\n    swap_count = self.execute_algorithm(\n  File \"/home/jmsb00nd/Documents/auto-heuristics/src/mapping/routing.py\", line 151, in execute_algorithm\n    local_swap_count = self.apply_qlosure_heuristic(param)\n  File \"/home/jmsb00nd/Documents/auto-heuristics/src/mapping/routing.py\", line 231, in apply_qlosure_heuristic\n    score = self.qlosure_poly_heuristic(swap_gate)\n  File \"<string>\", line 10, in qlosure_poly_heuristic\n  File \"<string>\", line 10, in <genexpr>\nIndexError: list index out of range\n"}

def qlosure_poly_heuristic(self, swap_gate):
    # --- Topology-adaptive beta (cached) ----------------------------------
    # Normalise the quartic term so that at the graph diameter the quadratic
    # and quartic contributions are equal: beta = 1 / diameter^2.
    # This makes the potential symmetric at the extremes and strictly steeper
    # than pure d^2 for all d > 1, without blowing up on small graphs.
    if not hasattr(self, '_apwc_beta'):
        dm = self.distance_matrix
        n  = self.num_qubits
        diameter = max(dm[i][j] for i in range(n) for j in range(n))
        # Guard against degenerate topologies
        self._apwc_beta = 1.0 / (diameter * diameter) if diameter > 0 else 0.1

    beta = self._apwc_beta

    # Inline hot-path lookups
    tm   = self.temp_mapping_dict
    dm   = self.distance_matrix
    a2q  = self.access2q
    ddc  = self.dag_dependencies_count
    eli  = self.extended_layer_index

    # ------------------------------------------------------------------
    # Anharmonic potential V(d) = d^2 + beta * d^4
    # For d=1: V≈1+beta   (barely penalised)
    # For d=4: V=16+256*beta  (steeply penalised – the quartic "wall")
    # ------------------------------------------------------------------

    # --- Front-layer cost -----------------------------------------------
    # Each ready gate contributes its full anharmonic potential,
    # weighted by criticality (downstream dependency count + 1).
    f_cost = 0.0
    fl = self.front_layer
    fl_size = len(fl)
    if fl_size == 0:
        return 0.0

    for gid in fl:
        qs = a2q[gid]
        p1, p2 = tm[qs[0]], tm[qs[1]]
        if p1 == -1 or p2 == -1:
            continue
        d = dm[p1][p2]
        d2 = d * d
        V = d2 + beta * d2 * d2          # d^2 + beta*d^4
        crit = ddc[gid] + 1
        f_cost += crit * V

    # --- Extended-layer (lookahead) cost --------------------------------
    # Future gates discount linearly with temporal depth so that the
    # immediate horizon dominates.  A tighter lookahead cap (20 gates)
    # prevents timeout on deep circuits.
    e_cost = 0.0
    el = self.extended_layer
    el_size = len(el)

    MAX_LOOKAHEAD = 20
    seen = 0
    for gid in el:
        if seen >= MAX_LOOKAHEAD:
            break
        qs = a2q[gid]
        p1, p2 = tm[qs[0]], tm[qs[1]]
        if p1 == -1 or p2 == -1:
            seen += 1
            continue
        d = dm[p1][p2]
        d2 = d * d
        V = d2 + beta * d2 * d2
        depth = eli.get(gid, 0) + 1      # 1-indexed depth
        crit  = ddc[gid] + 1
        e_cost += crit * V / depth
        seen += 1

    # --- Qubit-health multiplier ----------------------------------------
    # Prefer swaps that avoid hot/noisy physical qubits.
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # --- Aggregate ----------------------------------------------------------
    # Normalise each layer by its own size so circuits with large front
    # layers don't swamp the lookahead signal.
    H = max_decay * (
        f_cost / fl_size
        + (0.5 * e_cost / el_size if el_size else 0.0)
    )

    return H