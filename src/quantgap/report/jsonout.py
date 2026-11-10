from __future__ import annotations

import re
from pathlib import Path

from quantgap.models import CompareReport, ScoreResult


def safe_name(r: ScoreResult) -> str:
    base = f"{(r.pack_name or r.pack_id).split('@')[0]}-{r.quant.label}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def write_result(r: ScoreResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(r.to_json() + "\n", encoding="utf-8")
    return p


def write_report(rep: CompareReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(rep.model_dump_json(indent=2, by_alias=True) + "\n", encoding="utf-8")
    return p
