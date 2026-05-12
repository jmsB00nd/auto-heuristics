"""Per-run plot of cumulative LLM tokens vs. heuristic quality metrics."""
from __future__ import annotations

import os
from typing import Iterable, List, Dict


def plot_tokens_vs_metrics(trace: Iterable[Dict], out_path: str, run_id: str = "") -> None:
    """Render a 2-row figure: swaps (top) and depth (bottom) vs cumulative tokens.
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

    def _stage_key(stage: str) -> str:
        return "implementation" if stage == "implementation" else "refinement"

    colors = {"implementation": "#1f77b4", "refinement": "#ff7f0e"}

    fig, (ax_swaps, ax_depth) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    if successes:
        successes.sort(key=lambda r: r["cumulative_total_tokens"])
        xs = [r["cumulative_total_tokens"] for r in successes]
        swaps = [r["mean_swaps"] for r in successes]
        depths = [r["mean_depth"] for r in successes]
        stage_colors = [colors[_stage_key(r["stage"])] for r in successes]

        ax_swaps.plot(xs, swaps, color="#999999", linewidth=1, zorder=1)
        ax_depth.plot(xs, depths, color="#999999", linewidth=1, zorder=1)

        ax_swaps.scatter(xs, swaps, c=stage_colors, s=55, zorder=2, edgecolors="black", linewidths=0.4)
        ax_depth.scatter(xs, depths, c=stage_colors, s=55, zorder=2, edgecolors="black", linewidths=0.4)

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
        plt.Line2D([0], [0], marker="o", color="w", label="implementation",
                   markerfacecolor=colors["implementation"], markeredgecolor="black", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", label="refinement",
                   markerfacecolor=colors["refinement"], markeredgecolor="black", markersize=8),
    ]
    if failures:
        legend_handles.append(
            plt.Line2D([0], [0], marker="x", color="#888888", linestyle="", label="eval failed", markersize=8)
        )

    ax_swaps.set_ylabel("Mean swaps")
    ax_depth.set_ylabel("Mean depth")
    ax_depth.set_xlabel("Cumulative tokens consumed (input + output)")
    ax_swaps.grid(True, alpha=0.3)
    ax_depth.grid(True, alpha=0.3)
    ax_swaps.legend(handles=legend_handles, loc="best", fontsize=9)

    title = f"Tokens vs. heuristic quality — {run_id}" if run_id else "Tokens vs. heuristic quality"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
