"""Build the static leaderboard from a folder of result files.

    python site/build.py --results ./results --out site/dist

`--results` is a checkout of the results dataset: `results/<pack_id>/<sha256>.json`.
Output: index.html, one page per pack (model), data.json, app.js, style.css, logo.svg.
No framework and no build tools: plain HTML, a little JavaScript for sorting and the chart.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from string import Template

from pydantic import BaseModel

from quantgap import __version__
from quantgap.models import ScoreResult
from quantgap.util import human_size, utc_now

HERE = Path(__file__).parent
SITE_SCHEMA = "quantgap.site/1"
REPO_URL = "https://github.com/saqlain2204/quantgap"


class PackEntry(BaseModel):
    pack_id: str
    pack_name: str
    model: str
    results: list[ScoreResult]


class SiteData(BaseModel):
    schema_: str = SITE_SCHEMA
    generated_at: str
    quantgap: str
    packs: list[PackEntry]

    model_config = {"populate_by_name": True}


def load_results(root: Path) -> tuple[list[ScoreResult], list[str]]:
    """Read every result file. Bad files are skipped and reported, never fatal."""
    results: dict[tuple[str, str], ScoreResult] = {}
    problems: list[str] = []
    for f in sorted(root.rglob("*.json")):
        try:
            r = ScoreResult.model_validate(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            problems.append(f"{f}: {str(e).splitlines()[0]}")
            continue
        if r.quick or not r.quant.sha256:
            problems.append(f"{f}: quick or unhashed result, skipped")
            continue
        key = (r.pack_id, r.quant.sha256)
        if key not in results or r.created_at > results[key].created_at:
            results[key] = r
    return list(results.values()), problems


def group_by_pack(results: list[ScoreResult]) -> list[PackEntry]:
    packs: dict[str, list[ScoreResult]] = {}
    for r in results:
        packs.setdefault(r.pack_id, []).append(r)
    out = []
    for pid, rs in packs.items():
        name = rs[0].pack_name or pid
        out.append(
            PackEntry(
                pack_id=pid,
                pack_name=name,
                model=name.split("@")[0],
                results=sorted(rs, key=lambda r: r.metrics.kld_mean),
            )
        )
    return sorted(out, key=lambda p: p.model.lower())


def _e(x: object) -> str:
    return html.escape(str(x), quote=True)


def _row(r: ScoreResult) -> str:
    m, d = r.metrics, r.metrics.damage
    marker = (
        ' <abbr class="approx" title="approximate (server top-N)">approx</abbr>'
        if r.approximate
        else ""
    )
    nd = ' <abbr class="approx" title="non-deterministic">nd</abbr>' if r.nondeterministic else ""
    cells = [
        (f"{_e(r.quant.label)}{marker}{nd}", r.quant.label.lower()),
        (_e(r.quant.publisher or "-"), (r.quant.publisher or "").lower()),
        (_e(human_size(r.quant.size_bytes)), r.quant.size_bytes),
        (f"{m.kld_mean:.4f}", m.kld_mean),
        (f"{m.kld_mean_ci95[0]:.4f} to {m.kld_mean_ci95[1]:.4f}", m.kld_mean_ci95[0]),
        (f"{m.kld_p99:.3f}", m.kld_p99),
        (f"{m.top1_agree * 100:.1f}%", m.top1_agree),
        (f"{m.ppl_ratio:.3f}", m.ppl_ratio),
        (f"{d.loops}/{d.loops_ref}" if d else "-", d.loops if d else -1),
        (f"{d.wrong_script}" if d else "-", d.wrong_script if d else -1),
        (f"{d.json_valid}/{d.json_n}" if d else "-", d.json_valid if d else -1),
        (_e(r.backend.name), r.backend.name),
    ]
    tds = "".join(f'<td data-v="{_e(v)}">{c}</td>' for c, v in cells)
    return f'<tr data-sha="{_e(r.quant.sha256)}">{tds}</tr>'


HEADERS = ["Quant", "Publisher", "Size", "KLD mean", "95% interval", "Worst 1%", "Top-1", "PPL ratio",
           "Loops (q/ref)", "Wrong script", "JSON ok", "Backend"]  # fmt: skip


def render_model(tpl: Template, p: PackEntry) -> str:
    head = "".join(f'<th scope="col" tabindex="0" role="button">{h}</th>' for h in HEADERS)
    rows = "\n".join(_row(r) for r in p.results)
    chart_data = [
        {
            "label": r.quant.label, "size": r.quant.size_bytes, "kld": r.metrics.kld_mean,
            "lo": r.metrics.kld_mean_ci95[0], "hi": r.metrics.kld_mean_ci95[1],
            "approx": r.approximate,
        }
        for r in p.results
    ]  # fmt: skip
    return tpl.substitute(
        title=_e(p.model), pack_name=_e(p.pack_name), pack_id=_e(p.pack_id), n=len(p.results),
        head=head, rows=rows, version=_e(__version__), repo=REPO_URL,
        chart_json=json.dumps(chart_data).replace("</", "<\\/"),
    )  # fmt: skip


def render_index(tpl: Template, packs: list[PackEntry]) -> str:
    items = []
    for p in packs:
        best = p.results[0]
        items.append(
            f'<tr><td><a href="model-{_e(p.pack_id)}.html">{_e(p.model)}</a></td>'
            f"<td>{len(p.results)}</td>"
            f"<td>{_e(best.quant.label)} ({best.metrics.kld_mean:.4f})</td>"
            f"<td><code>{_e(p.pack_id)}</code></td></tr>"
        )
    total = sum(len(p.results) for p in packs)
    body = (
        "\n".join(items)
        or '<tr><td colspan="4">No results yet. The leaderboard opens with the first full results: <code>quantgap submit</code>.</td></tr>'
    )
    return tpl.substitute(
        rows=body, n_models=len(packs), n_results=total, version=_e(__version__), repo=REPO_URL,
        generated=_e(utc_now()),
    )  # fmt: skip


def build(results_dir: Path, out: Path) -> SiteData:
    results, problems = load_results(results_dir) if results_dir.exists() else ([], [])
    for msg in problems:
        print(f"warning: {msg}", file=sys.stderr)
    packs = group_by_pack(results)
    data = SiteData(generated_at=utc_now(), quantgap=__version__, packs=packs)

    out.mkdir(parents=True, exist_ok=True)
    for f in (HERE / "static").iterdir():
        shutil.copy(f, out / f.name)
    logo = HERE.parent / "assets" / "logo.svg"
    if logo.exists():
        shutil.copy(logo, out / "logo.svg")
    (out / "data.json").write_text(
        json.dumps(data.model_dump(by_alias=True, mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    model_tpl = Template((HERE / "templates" / "model.html").read_text(encoding="utf-8"))
    index_tpl = Template((HERE / "templates" / "index.html").read_text(encoding="utf-8"))
    (out / "index.html").write_text(render_index(index_tpl, packs), encoding="utf-8")
    for p in packs:
        (out / f"model-{p.pack_id}.html").write_text(render_model(model_tpl, p), encoding="utf-8")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", type=Path)
    ap.add_argument("--out", default=HERE / "dist", type=Path)
    a = ap.parse_args()
    data = build(a.results, a.out)
    print(
        f"built {len(data.packs)} model page(s), {sum(len(p.results) for p in data.packs)} result(s) -> {a.out}"
    )


if __name__ == "__main__":
    main()
