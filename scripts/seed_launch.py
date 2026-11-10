"""Score a list of quants for each pack and collect the result files (milestone M7).

    python scripts/seed_launch.py scripts/seed_list.json --out seed-results

The seed list is JSON:

    {"packs": [
        {"pack": "packs/Qwen2.5-0.5B-Instruct",
         "quants": ["hf://bartowski/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"]}
    ]}

A `quants` entry may also be {"repo": "org/repo", "glob": "*Q4*.gguf"}. Results are written one
file per quant, and a comparison chart plus table per pack. Nothing is invented: if a quant fails,
the error is printed and the run goes on with the next one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def expand(entry: str | dict) -> list[str]:
    if isinstance(entry, str):
        return [entry]
    from quantgap.hub.resolve import expand_glob

    return [f"hf://{entry['repo']}/{f}" for f in expand_glob(entry["repo"], entry["glob"])]


def result_name(quant: str) -> str:
    """One file per quant. The publisher is part of the name: two publishers often ship files whose
    names differ only in letter case (Q4_K_M), which would collide on Windows and macOS."""
    parts = quant.removeprefix("hf://").split("/")
    org = parts[0] if quant.startswith("hf://") and len(parts) > 2 else "local"
    return f"{org}__{Path(quant).stem}.json".lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("seed_list", type=Path)
    ap.add_argument("--out", type=Path, default=Path("seed-results"))
    ap.add_argument("--quick", action="store_true", help="first look only, not submittable")
    ap.add_argument("--n-gpu-layers", type=int, default=None)
    a = ap.parse_args()

    spec = json.loads(a.seed_list.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for p in spec["packs"]:
        name = Path(p["pack"]).name
        quants = [q for entry in p["quants"] for q in expand(entry)]
        for q in quants:
            out = a.out / name / result_name(q)
            if out.exists():
                print(f"skip (done): {q}")
                continue
            cmd = [
                sys.executable,
                "-m",
                "quantgap.cli",
                "score",
                q,
                "--pack",
                p["pack"],
                "--json",
                str(out),
            ]
            if a.quick:
                cmd.append("--quick")
            if a.n_gpu_layers is not None:
                cmd += ["--n-gpu-layers", str(a.n_gpu_layers)]
            print("->", q, flush=True)
            if subprocess.run(cmd).returncode != 0:
                failures += 1
                print(f"FAILED: {q}", file=sys.stderr)
        results = sorted(f for f in (a.out / name).glob("*.json") if f.name != "report.json")
        if results:
            subprocess.run(
                [sys.executable, "-m", "quantgap.cli", "compare", *map(str, results),
                 "--chart", str(a.out / name / "chart.svg"), "--json", str(a.out / name / "report.json")]
            )  # fmt: skip
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
