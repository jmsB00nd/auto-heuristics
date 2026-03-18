def qlosure_poly_heuristic(self, swap_gate):
    """
    Qubit Momentum Reversal Penalty (QMRP) heuristic.

    Core idea: maintain a per-physical-qubit "committed distance" history.
    After each round, record curr_dist[p] = distance from physical qubit p
    to its nearest front-layer target under the committed mapping. When
    evaluating a candidate SWAP, check whether either swapped qubit was
    previously approaching its target (prev_dist[p] > curr_dist[p]) but
    the candidate would reverse that progress (new_dist[p] > curr_dist[p]).
    The reversal penalty is the sum of reversal magnitudes across both qubits,
    added as a soft penalty to break ties and suppress ping-pong loops.

    Structure:
      H = max_decay * (f_score + W * e_score + ALPHA * reversal_penalty)

    where:
      f_score  = dependency-weighted avg distance over front_layer     [dominates]
      e_score  = depth-discounted dependency-weighted avg over extended_layer [W < 1]
      reversal_penalty = sum of (new_d - curr_d) for qubits with positive
                         momentum that the candidate SWAP would reverse    [tie-break]
    """
    W     = 0.5   # Extended layer weight: must be < 1 to keep FL dominant
    ALPHA = 0.5   # Reversal penalty weight: tie-breaking scale

    p1, p2 = swap_gate
    front_layer_size    = len(self.front_layer)
    extended_layer_size = len(self.extended_layer)
    max_decay = max(self.decay_parameter[p1], self.decay_parameter[p2])

    # ================================================================
    # QMRP: Round-boundary detection
    #
    # A "round" is uniquely identified by BOTH the committed mapping AND
    # the current front layer:
    #   - mapping changes     -> a SWAP was just committed
    #   - front_layer changes -> gates just executed, new FL gates appeared
    # Both events require the per-round cache to be refreshed.
    # ================================================================
    round_key = (hash(tuple(self.mapping_dict)), hash(frozenset(self.front_layer)))

    if not hasattr(self, '_qmrp_round_key') or self._qmrp_round_key != round_key:
        # Transition: save current distances as the "momentum baseline" for
        # the next round, then reset the current-round cache.
        self._qmrp_prev_dist  = getattr(self, '_qmrp_curr_dist', {})
        self._qmrp_curr_dist  = {}
        self._qmrp_round_key  = round_key
        # Cache the physical positions of all FL gate target qubits under the
        # committed mapping.  Stable for every candidate in this round.
        self._qmrp_fl_targets = [
            self.mapping_dict[q]
            for g in self.front_layer
            for q in self.access2q[g]
        ]

    fl_targets = self._qmrp_fl_targets  # list[int], physical qubit positions

    # ================================================================
    # QMRP: Reversal penalty
    #
    # For each of the two swapped physical qubits p in {p1, p2}:
    #   curr_d = distance from p to nearest FL target (committed mapping)
    #   prev_d = same metric from the PREVIOUS round's committed mapping
    #   new_d  = same metric if this candidate SWAP were committed
    #
    # Reversal condition: qubit had positive momentum (prev_d > curr_d,
    # i.e., it was getting closer to a target) AND the candidate would
    # undo that progress (new_d > curr_d).
    #
    # Penalty magnitude = new_d - curr_d  (extra hops the reversal costs)
    # ================================================================
    reversal_penalty = 0.0

    if fl_targets:
        # Precompute FL targets under the proposed mapping once per candidate.
        # Differs from fl_targets only if a swapped logical qubit is a FL endpoint.
        temp_fl_targets = [
            self.temp_mapping_dict[q]
            for g in self.front_layer
            for q in self.access2q[g]
        ]

        for p in (p1, p2):
            # curr_dist is lazily memoised for the lifetime of this round.
            if p not in self._qmrp_curr_dist:
                self._qmrp_curr_dist[p] = min(
                    self.distance_matrix[p][t] for t in fl_targets
                )
            curr_d = self._qmrp_curr_dist[p]

            # prev_d defaults to curr_d when no history exists (first round or
            # after a gate-execution reset).  Default -> penalty is zero: neutral.
            prev_d = self._qmrp_prev_dist.get(p, curr_d)

            # Distance from p to nearest FL target under the proposed mapping.
            new_d = min(self.distance_matrix[p][t] for t in temp_fl_targets)

            # Only penalise confirmed reversals: momentum was positive AND
            # the candidate undoes it.  Magnitude = size of the backward step.
            if prev_d > curr_d and new_d > curr_d:
                reversal_penalty += float(new_d - curr_d)

    # ================================================================
    # Base cost -- Front Layer (strict dominance enforced by W < 1)
    # ================================================================
    f_distance = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        deps = self.dag_dependencies_count[g]
        f_distance += (deps + 1) * self.distance_matrix[Q1][Q2]

    f_score = f_distance / front_layer_size if front_layer_size else 0.0

    # ================================================================
    # Extended Layer (depth-discounted lookahead)
    # ================================================================
    e_distance = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1 = self.temp_mapping_dict[q1]
        Q2 = self.temp_mapping_dict[q2]
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps = self.dag_dependencies_count[g]
        e_distance += (deps + 1) * self.distance_matrix[Q1][Q2] / layer_factor

    e_score = e_distance / extended_layer_size if extended_layer_size else 0.0

    # ================================================================
    # Final cost
    # ================================================================
    H = max_decay * (f_score + W * e_score + ALPHA * reversal_penalty)
    return H