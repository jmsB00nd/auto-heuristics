# Idea: \_NAME: Fourier Topology Frequency Cost (FTFC)
# Stats: {"mean_swaps": Infinity, "mean_depth": 0, "mean_runtime": 0, "total_circuits": 22, "successful_runs": 0, "failed_runs": 3, "error": "All circuits failed", "first_failure_error": "list index out of range", "first_failure_traceback": "Traceback (most recent call last):\n  File \"/home/jmsb00nd/Documents/auto-heuristics/orchestrator.py\", line 228, in inject_and_run\n    raise exception_container['error']\n  File \"/home/jmsb00nd/Documents/auto-heuristics/orchestrator.py\", line 210, in run_with_timeout\n    min_swaps, min_depth, _ = router.run(heuristic_method=\"Qlosure\")\n  File \"/home/jmsb00nd/Documents/auto-heuristics/src/mapping/routing.py\", line 102, in run\n    swap_count = self.execute_algorithm(\n  File \"/home/jmsb00nd/Documents/auto-heuristics/src/mapping/routing.py\", line 151, in execute_algorithm\n    local_swap_count = self.apply_qlosure_heuristic(param)\n  File \"/home/jmsb00nd/Documents/auto-heuristics/src/mapping/routing.py\", line 231, in apply_qlosure_heuristic\n    score = self.qlosure_poly_heuristic(swap_gate)\n  File \"<string>\", line 23, in qlosure_poly_heuristic\nIndexError: list index out of range\n"}

def qlosure_poly_heuristic(self, swap_gate):
    import numpy as np

    N = self.num_qubits
    max_decay = max(
        self.decay_parameter[swap_gate[0]],
        self.decay_parameter[swap_gate[1]]
    )

    # ------------------------------------------------------------------ #
    #  Build & cache the frequency-domain distance matrix (one-time cost) #
    # ------------------------------------------------------------------ #
    if not hasattr(self, '_ftfc_dist'):

        # Step 1 – smooth distance-decay signal for every physical qubit.
        # s_q[j] = 1/(d(q,j)+1)  maps graph distances to a "soft influence
        # field".  Treating qubit indices as positions in a periodic 1-D
        # domain is the key Fourier-domain embedding.
        signals = np.empty((N, N), dtype=np.float64)
        for q in range(N):
            row = self.distance_matrix[q]
            for j in range(N):
                signals[q, j] = 1.0 / (row[j] + 1.0)

        # Step 2 – row-wise DFT; spectra[q, k] = k-th Fourier mode of q's
        # signal.  This is where topology periodicity is encoded.
        spectra = np.fft.fft(signals, axis=1)          # (N, N) complex
        mag     = np.abs(spectra)                       # (N, N) real

        # Step 3 – vectorised pairwise spectral phase-incoherence.
        #
        # DFT_dist(Q1,Q2) = Σ_k |F^Q1_k|·|F^Q2_k|·(1−cos Δφ_k)
        #                 / Σ_k |F^Q1_k|·|F^Q2_k|
        #
        # Re-expressed in matrix form (avoids triple loop):
        #   cross_power[q1,q2]    = Σ_k mag[q1,k]·mag[q2,k]   = mag @ mag.T
        #   coherent_power[q1,q2] = Re(Σ_k F^q1_k · conj(F^q2_k))
        #                         = Re(spectra @ spectra^H)
        # numerator = cross_power − coherent_power
        #           = Σ_k |F^Q1_k|·|F^Q2_k|·(1−cos Δφ_k)  ≥ 0
        # Both operations reduce to BLAS matrix multiplies → computed once.

        cross_power    = mag @ mag.T                           # (N, N) real
        coherent_power = np.real(spectra @ spectra.conj().T)   # (N, N) real

        numerator   = cross_power - coherent_power             # ≥ 0
        denominator = cross_power + 1e-10

        ftfc = numerator / denominator                         # ∈ [0, 1]
        np.fill_diagonal(ftfc, 0.0)
        self._ftfc_dist = ftfc

    ftfc = self._ftfc_dist

    front_layer_size    = max(len(self.front_layer), 1)
    extended_layer_size = len(self.extended_layer)

    # Front layer – immediate gates weighted by DAG criticality
    f_cost = 0.0
    for g in self.front_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 == -1 or Q2 == -1:
            continue
        deps    = self.dag_dependencies_count[g]
        f_cost += (deps + 1) * ftfc[Q1, Q2]

    # Extended layer – lookahead discounted by depth
    e_cost = 0.0
    for g in self.extended_layer:
        q1, q2 = self.access2q[g]
        Q1, Q2 = self.temp_mapping_dict[q1], self.temp_mapping_dict[q2]
        if Q1 == -1 or Q2 == -1:
            continue
        layer_factor = self.extended_layer_index.get(g, 0) + 1
        deps         = self.dag_dependencies_count[g]
        e_cost      += (deps + 1) * ftfc[Q1, Q2] / layer_factor

    H = max_decay * (
        f_cost / front_layer_size
        + (e_cost / extended_layer_size if extended_layer_size else 0.0)
    )

    return H