"""Quality-vs-size chart. One series (the quants), one hue, direct labels, no legend box.

Lower and further left is better: a smaller file with less drift. The line joins the quants no
other quant beats on both size and KLD (the "best for its size" frontier).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from quantgap.models import ScoreResult  # noqa: E402

INK = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e7eb"
MARK = "#2563eb"
SURFACE = "#ffffff"
FLAG = "#b45309"


def pareto(results: list[ScoreResult]) -> list[ScoreResult]:
    """Quants not beaten by a smaller-or-equal file with lower-or-equal KLD."""
    out: list[ScoreResult] = []
    best = float("inf")
    for r in sorted(results, key=lambda r: (r.quant.size_bytes, r.metrics.kld_mean)):
        if r.metrics.kld_mean < best:
            out.append(r)
            best = r.metrics.kld_mean
    return out


def draw_chart(
    results: list[ScoreResult],
    out: str | Path,
    title: str | None = None,
    budget_bytes: int | None = None,
) -> Path:
    out = Path(out)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    rs = sorted(results, key=lambda r: r.quant.size_bytes)
    xs = [r.quant.size_bytes / 1e9 for r in rs]
    ys = [r.metrics.kld_mean for r in rs]
    lo = [r.metrics.kld_mean - r.metrics.kld_mean_ci95[0] for r in rs]
    hi = [r.metrics.kld_mean_ci95[1] - r.metrics.kld_mean for r in rs]
    ax.errorbar(
        xs, ys, yerr=[[max(0, v) for v in lo], [max(0, v) for v in hi]],
        fmt="none", ecolor=MARK, elinewidth=1, alpha=0.45, capsize=0,
    )  # fmt: skip
    front = pareto(rs)
    ax.plot(
        [r.quant.size_bytes / 1e9 for r in front], [r.metrics.kld_mean for r in front],
        color=MARK, linewidth=1.5, alpha=0.5, zorder=2,
    )  # fmt: skip
    ax.scatter(xs, ys, s=64, color=MARK, edgecolor=SURFACE, linewidth=2, zorder=3)
    for r, x, y in zip(rs, xs, ys, strict=True):
        label = r.quant.label + (" *" if r.approximate else "")
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=8, color=INK)
    if budget_bytes:
        ax.axvline(budget_bytes / 1e9, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            "memory budget", (budget_bytes / 1e9, 1), xycoords=("data", "axes fraction"),
            xytext=(-4, -12), textcoords="offset points", ha="right", fontsize=8, color=MUTED,
        )  # fmt: skip
    ax.set_xlabel("File size (GB)", color=MUTED, fontsize=9)
    ax.set_ylabel("KLD mean vs original (lower is better)", color=MUTED, fontsize=9)
    if min(ys) > 0 and max(ys) / max(min(ys), 1e-12) > 30:
        ax.set_yscale("log")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    pack = results[0].pack_name or results[0].pack_id
    ax.set_title(title or f"Quality vs size  |  pack {pack}", loc="left", color=INK, fontsize=11)
    if any(r.approximate for r in rs):
        fig.text(0.01, 0.01, "* approximate (server top-N)", fontsize=7, color=FLAG)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out
