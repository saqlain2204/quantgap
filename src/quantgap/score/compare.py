"""Build a comparison report from several results (one pack only)."""

from __future__ import annotations

from quantgap.models import CompareReport, ScoreResult
from quantgap.score.recommend import ASSUMPTION, recommend


class MixedPacksError(ValueError):
    pass


def build_report(results: list[ScoreResult], budget_bytes: int | None = None) -> CompareReport:
    if not results:
        raise ValueError("Nothing to compare.")
    ids = {r.pack_id for r in results}
    if len(ids) > 1:
        raise MixedPacksError(
            "These results come from different packs, so their numbers cannot share one "
            f"table: {', '.join(sorted(ids))}. Score all quants against the same pack."
        )
    rec_text = None
    if budget_bytes:
        rec = recommend(results, budget_bytes)
        rec_text = " ".join([rec.sentence, *rec.warnings])
    return CompareReport(
        pack_id=results[0].pack_id,
        pack_name=results[0].pack_name,
        results=results,
        budget_bytes=budget_bytes,
        recommendation=rec_text,
        assumptions=ASSUMPTION if budget_bytes else None,
    )
