"""Rich tables for the terminal."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from quantgap.models import ScoreResult
from quantgap.util import human_size


def print_result(c: Console, r: ScoreResult) -> None:
    m = r.metrics
    c.print(f"[bold]quantgap {r.quantgap}[/bold]  pack: {r.pack_name}")
    flags = " ".join(
        f"[yellow]{t}[/yellow]"
        for t, on in (
            ("quick", r.quick),
            ("approximate", r.approximate),
            ("non-deterministic", r.nondeterministic),
        )
        if on
    )
    c.print(
        f"quant: {r.quant.source}  [bold]{r.quant.label}[/bold]  {human_size(r.quant.size_bytes)}  "
        f"{r.backend.name}  {flags}"
    )
    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_column("note")
    t.add_row(
        "KLD mean",
        f"{m.kld_mean:.4f}",
        f"95% CI [{m.kld_mean_ci95[0]:.4f}, {m.kld_mean_ci95[1]:.4f}]  (lower is better)",
    )
    t.add_row("KLD worst 1%", f"{m.kld_p99:.3f}", "99th percentile of positions")
    t.add_row("KLD worst 0.1%", f"{m.kld_p999:.3f}", "99.9th percentile")
    t.add_row("top-1 agree", f"{m.top1_agree * 100:.1f} %", "same best guess as the original")
    t.add_row("ppl ratio", f"{m.ppl_ratio:.4f}", "1.0 means no change")
    d = m.damage
    if d:
        t.add_row("loops", f"{d.loops} / {d.n_prompts}", f"original: {d.loops_ref}")
        t.add_row(
            "wrong script", f"{d.wrong_script} / {d.n_prompts}", "script not in the original's text"
        )
        t.add_row("valid JSON", f"{d.json_valid} / {d.json_n}", f"original: {d.json_valid_ref}")
        t.add_row(
            "first divergence",
            f"{d.first_divergence_median:.0f} tokens",
            "median, vs the original's greedy text",
        )
    c.print(t)
    c.print("by group:  " + "  ".join(f"{g} {gm.kld_mean:.4f}" for g, gm in m.by_group.items()))
    for w in r.warnings:
        c.print(f"[yellow]note:[/yellow] {w}")


def print_compare(c: Console, results: list[ScoreResult]) -> None:
    t = Table(show_header=True, header_style="bold")
    for col, j in (("quant", "left"), ("size", "right"), ("KLD mean", "right"), ("95% CI", "right"),
                   ("worst 1%", "right"), ("top-1", "right"), ("ppl", "right"), ("loops", "right"),
                   ("wrong script", "right"), ("JSON ok", "right")):  # fmt: skip
        t.add_column(col, justify=j)
    for r in sorted(results, key=lambda r: r.metrics.kld_mean):
        m, d = r.metrics, r.metrics.damage
        t.add_row(
            r.quant.label + (" *" if r.approximate else "") + (" (quick)" if r.quick else ""),
            human_size(r.quant.size_bytes),
            f"{m.kld_mean:.4f}",
            f"{m.kld_mean_ci95[0]:.4f}-{m.kld_mean_ci95[1]:.4f}",
            f"{m.kld_p99:.3f}",
            f"{m.top1_agree * 100:.1f}%",
            f"{m.ppl_ratio:.3f}",
            f"{d.loops}/{d.loops_ref}" if d else "-",
            f"{d.wrong_script}" if d else "-",
            f"{d.json_valid}/{d.json_n}" if d else "-",
        )
    c.print(t)
    if any(r.approximate for r in results):
        c.print("* approximate (server top-N)")
    c.print("loops shows quant/original. Lower KLD is better.")
