"""Per-run plot of cumulative LLM tokens vs. heuristic quality metrics.

Evaluated heuristics are grouped into the pipeline's two phases:

* **Ideation**  — first implementation of generated ideas + every re-ideation
  cycle (stages ``implementation`` and ``reideation_*``).
* **Evolution** — crossover children and elite mutants
  (stages ``evolution/iter_*/crossover`` and ``.../mutation``).

The phases interleave along the token axis (impl → evolve → re-ideate → evolve
→ …), so each evaluated point is colored by its phase and a faint background
band marks which phase the run was in at that token count.
"""
from __future__ import annotations

import os
from typing import Iterable, List, Dict


PHASE_COLORS = {"ideation": "#1f77b4", "evolution": "#ff7f0e"}
PHASE_LABELS = {
    "ideation": "Ideation (init + re-ideation)",
    "evolution": "Evolution (crossover + mutation)",
}


def _phase(stage: str) -> str:
    """Map an eval-trace ``stage`` string to one of the two pipeline phases."""
    s = (stage or "").lower()
    if s == "implementation" or s.startswith("reideation"):
        return "ideation"
    if "crossover" in s or "mutation" in s or s.startswith("evolution"):
        return "evolution"
    return "ideation"


def plot_tokens_vs_metrics(trace: Iterable[Dict], out_path: str, run_id: str = "") -> None:
    """Render a 2-row figure: swaps (top) and depth (bottom) vs cumulative tokens,
    with points and background bands separated by pipeline phase.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records: List[Dict] = [dict(r) for r in trace]
    if not records:
        return

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    successes = [r for r in records if not r.get("error")]
    failures = [r for r in records if r.get("error")]

    fig, (ax_swaps, ax_depth) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    # --- Background phase bands (drawn from ALL records along the token axis) ---
    all_recs = sorted(records, key=lambda r: r["cumulative_total_tokens"])
    xs_all = [float(r["cumulative_total_tokens"]) for r in all_recs]
    phases_all = [_phase(r.get("stage", "")) for r in all_recs]
    pad = ((xs_all[-1] - xs_all[0]) or 1.0) * 0.02
    for i, x in enumerate(xs_all):
        left = (xs_all[i - 1] + x) / 2 if i > 0 else x - pad
        right = (x + xs_all[i + 1]) / 2 if i < len(xs_all) - 1 else x + pad
        band = PHASE_COLORS[phases_all[i]]
        for ax in (ax_swaps, ax_depth):
            ax.axvspan(left, right, color=band, alpha=0.08, zorder=0, linewidth=0)

    if successes:
        successes.sort(key=lambda r: r["cumulative_total_tokens"])
        xs = [r["cumulative_total_tokens"] for r in successes]
        swaps = [r["mean_swaps"] for r in successes]
        depths = [r["mean_depth"] for r in successes]
        point_colors = [PHASE_COLORS[_phase(r["stage"])] for r in successes]

        ax_swaps.plot(xs, swaps, color="#999999", linewidth=1, zorder=1)
        ax_depth.plot(xs, depths, color="#999999", linewidth=1, zorder=1)

        ax_swaps.scatter(xs, swaps, c=point_colors, s=55, zorder=2, edgecolors="black", linewidths=0.4)
        ax_depth.scatter(xs, depths, c=point_colors, s=55, zorder=2, edgecolors="black", linewidths=0.4)

        # Running-best markers
        best_idx_swap = int(min(range(len(swaps)), key=lambda i: swaps[i]))
        best_idx_depth = int(min(range(len(depths)), key=lambda i: depths[i]))
        ax_swaps.annotate(
            f"best: {swaps[best_idx_swap]:.2f}",
            xy=(xs[best_idx_swap], swaps[best_idx_swap]),
            xytext=(6, 6), textcoords="offset points", fontsize=9,
        )
        ax_depth.annotate(
            f"best: {depths[best_idx_depth]:.2f}",
            xy=(xs[best_idx_depth], depths[best_idx_depth]),
            xytext=(6, 6), textcoords="offset points", fontsize=9,
        )

    if failures:
        fx = [r["cumulative_total_tokens"] for r in failures]
        # Draw failures at the plot's top edge so they don't skew the y-scale.
        for ax in (ax_swaps, ax_depth):
            ymin, ymax = ax.get_ylim()
            yline = ymax if ymax > ymin else 1.0
            ax.scatter(fx, [yline] * len(fx), marker="x", c="#888888", s=45, label="eval failed")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=PHASE_LABELS["ideation"],
                   markerfacecolor=PHASE_COLORS["ideation"], markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label=PHASE_LABELS["evolution"],
                   markerfacecolor=PHASE_COLORS["evolution"], markeredgecolor="black", markersize=8),
    ]
    if failures:
        legend_handles.append(
            plt.Line2D([0], [0], marker="x", color="#888888", linestyle="", label="eval failed", markersize=8)
        )

    ax_swaps.set_ylabel("Mean swaps")
    ax_depth.set_ylabel("Mean depth")
    ax_depth.set_xlabel("Cumulative tokens consumed (words)")
    ax_swaps.grid(True, alpha=0.3)
    ax_depth.grid(True, alpha=0.3)
    ax_swaps.legend(handles=legend_handles, loc="best", fontsize=9)

    title = f"Tokens vs. heuristic quality — {run_id}" if run_id else "Tokens vs. heuristic quality"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
