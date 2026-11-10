"""Pick the best quant that fits a memory budget."""

from __future__ import annotations

from dataclasses import dataclass, field

from quantgap.models import ScoreResult
from quantgap.util import GIB, human_size

OVERHEAD_FACTOR = 1.1
OVERHEAD_BYTES = GIB
ASSUMPTION = (
    "Memory estimate = file size x 1.1 + 1 GiB (context buffers at 2K context). "
    "Real use depends on your engine and context length."
)
DAMAGE_WARN_GAP = 3  # prompts worse than the reference


def estimated_memory(size_bytes: int) -> float:
    return size_bytes * OVERHEAD_FACTOR + OVERHEAD_BYTES


def _overlap(a: ScoreResult, b: ScoreResult) -> bool:
    lo1, hi1 = a.metrics.kld_mean_ci95
    lo2, hi2 = b.metrics.kld_mean_ci95
    return lo1 <= hi2 and lo2 <= hi1


@dataclass
class Recommendation:
    sentence: str
    chosen: ScoreResult | None
    warnings: list[str] = field(default_factory=list)
    not_clearly_different: list[ScoreResult] = field(default_factory=list)


def _name(r: ScoreResult) -> str:
    return r.quant.label


def recommend(results: list[ScoreResult], budget_bytes: int) -> Recommendation:
    budget_txt = human_size(budget_bytes)
    fits = [r for r in results if estimated_memory(r.quant.size_bytes) <= budget_bytes]
    if not fits:
        smallest = min(results, key=lambda r: r.quant.size_bytes)
        need = estimated_memory(smallest.quant.size_bytes)
        return Recommendation(
            f"Nothing fits in {budget_txt}. The smallest, {_name(smallest)}, needs about "
            f"{human_size(need)}.",
            None,
        )
    best = min(fits, key=lambda r: r.metrics.kld_mean)
    ties = [r for r in fits if r is not best and _overlap(r, best)]
    pool = [best, *ties]
    chosen = min(pool, key=lambda r: (r.quant.size_bytes, r.metrics.kld_mean))
    size = human_size(chosen.quant.size_bytes)
    sentence = f"For {budget_txt}, use {_name(chosen)} ({size})."
    if chosen is not best:
        saved = best.quant.size_bytes - chosen.quant.size_bytes
        sentence += (
            f" It is not clearly different from {_name(best)}, which has the lowest KLD, "
            f"and is {human_size(saved)} smaller."
        )
    elif ties:
        names = ", ".join(_name(t) for t in ties)
        sentence += f" {names} are not clearly different but are larger."
    warnings: list[str] = []
    d = chosen.metrics.damage
    if d is not None:
        if d.loops - d.loops_ref >= DAMAGE_WARN_GAP:
            warnings.append(
                f"Warning: {_name(chosen)} loops in {d.loops} prompts, the original in {d.loops_ref}."
            )
        if d.wrong_script - d.wrong_script_ref >= DAMAGE_WARN_GAP:
            warnings.append(
                f"Warning: {_name(chosen)} writes the wrong script in {d.wrong_script} prompts."
            )
    return Recommendation(sentence, chosen, warnings, ties)
