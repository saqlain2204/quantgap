"""Probe set versions and their hashes. A probe set is frozen once released."""

from __future__ import annotations

import hashlib
from pathlib import Path

from quantgap.models import ProbeSeq

_DATA = Path(__file__).parent / "data"
AVAILABLE = {"v1": "probes_v1.jsonl"}


def probes_path(version: str = "v1") -> Path:
    if version not in AVAILABLE:
        raise ValueError(
            f"Unknown probe set {version!r}. Available: {', '.join(sorted(AVAILABLE))}."
        )
    return _DATA / AVAILABLE[version]


def probes_sha256(version: str = "v1") -> str:
    return hashlib.sha256(probes_path(version).read_bytes()).hexdigest()


def load_probes(version: str = "v1", path: Path | None = None) -> list[ProbeSeq]:
    p = path or probes_path(version)
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(ProbeSeq.model_validate_json(line))
    return rows
