"""Small helpers shared by several modules."""

from __future__ import annotations

import hashlib
import platform
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

_SIZE_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([kmgt]?i?b?)\s*$", re.IGNORECASE)
_UNITS = {
    "": 1,
    "b": 1,
    "k": 1000,
    "kb": 1000,
    "m": 1000**2,
    "mb": 1000**2,
    "g": 1000**3,
    "gb": 1000**3,
    "t": 1000**4,
    "tb": 1000**4,
    "ki": 1024,
    "kib": 1024,
    "mi": 1024**2,
    "mib": 1024**2,
    "gi": 1024**3,
    "gib": 1024**3,
    "ti": 1024**4,
    "tib": 1024**4,
}
GIB = 1024**3


def parse_size(text: str) -> int:
    """Parse '16GB', '16GiB', '16000MB' into bytes. GB means 10^9, GiB means 2^30."""
    m = _SIZE_RE.match(text)
    if not m or m.group(2).lower() not in _UNITS:
        raise ValueError(f"Cannot read size {text!r}. Try something like 16GB, 16GiB or 16000MB.")
    return int(float(m.group(1)) * _UNITS[m.group(2).lower()])


def human_size(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def hardware_string() -> str:
    """A short description of this machine, for the result file."""
    parts = [platform.processor() or platform.machine() or "unknown CPU"]
    try:  # optional, never required
        import torch

        if torch.cuda.is_available():
            parts = [torch.cuda.get_device_name(0) + " / CUDA"]
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            parts = [f"Apple {platform.machine()} / MPS"]
    except Exception:
        pass
    if platform.system() == "Darwin" and platform.machine() == "arm64" and "Apple" not in parts[0]:
        parts = ["Apple silicon / Metal"]
    return f"{parts[0]} ({platform.system()})"


def free_disk_bytes(path: str | Path) -> int:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free


def jsonable_float(x: float) -> float:
    """Round to 9 significant digits so result files are stable across runs."""
    return float(f"{x:.9g}")
