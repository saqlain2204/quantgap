"""quantgap command line: pack, packs, score, compare, submit, inspect, doctor."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from quantgap import __version__
from quantgap.backends import (
    BACKEND_NAMES,
    BackendSettings,
    ResolvedSource,
    get_backend,
    pick_backend_name,
)
from quantgap.config import Settings, load_settings
from quantgap.models import ScoreResult
from quantgap.util import parse_size

for _s in (sys.stdout, sys.stderr):  # Windows consoles default to a legacy code page
    with contextlib.suppress(AttributeError, ValueError):
        _s.reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    help="quantgap: how much did this quant lose? Measure how far a quantized model drifts "
    "from the original, on your own machine, without running the original.",
    no_args_is_help=True,
    add_completion=False,
)
packs_app = typer.Typer(help="List and inspect reference packs.", no_args_is_help=True)
app.add_typer(packs_app, name="packs")
console = Console()
err = Console(stderr=True)

OLLAMA_HINT = (
    "Ollama models cannot be scored by name (Ollama does not return prompt logprobs). "
    "Find the GGUF file Ollama already downloaded with `ollama show --modelfile NAME` "
    "(the FROM line is the path) and pass that file to quantgap."
)


def _fail(msg: str, code: int = 1) -> typer.Exit:
    err.print(f"[red]error:[/red] {escape(msg)}")
    return typer.Exit(code)


def _version(v: bool) -> None:
    if v:
        console.print(f"quantgap {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True)] = False,
) -> None:
    pass


# ---------------------------------------------------------------------------- helpers


def _settings_for(
    cfg: Settings, backend: str, ctx: int | None, n_gpu_layers: int | None,
    base_url: str | None, model_name: str | None,
) -> BackendSettings:  # fmt: skip
    return BackendSettings(
        ctx=ctx or cfg.defaults.ctx,
        n_gpu_layers=n_gpu_layers if n_gpu_layers is not None else cfg.llamacpp.n_gpu_layers,
        n_threads=cfg.llamacpp.n_threads,
        base_url=base_url or (cfg.openai.base_url if backend == "openai" else None),
        model_name=model_name or cfg.openai.model_name,
        max_prompt_logprobs=cfg.openai.max_prompt_logprobs,
    )


def _progress() -> tuple[Progress, callable]:
    prog = Progress(
        TextColumn("{task.description}"), BarColumn(), TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(), console=console, transient=True,
    )  # fmt: skip
    tasks: dict[str, int] = {}

    def cb(phase: str, done: int, total: int) -> None:
        if phase not in tasks:
            tasks[phase] = prog.add_task(phase, total=total)
        prog.update(tasks[phase], completed=done, total=total)

    return prog, cb


def _score_source(
    src: ResolvedSource, pack, cfg: Settings, *, backend_name: str, ctx, n_gpu_layers,
    base_url, model_name, groups, quick: bool, do_damage: bool,
) -> ScoreResult:  # fmt: skip
    from quantgap.hub.packs import fine_tune_warning
    from quantgap.score.run import score_quant

    name = backend_name if backend_name != "auto" else pick_backend_name(src, base_url)
    settings = _settings_for(cfg, name, ctx, n_gpu_layers, base_url, model_name)
    warns = []
    if (w := fine_tune_warning(src.base_model, pack)) is not None:
        warns.append(w)
    backend = get_backend(name)
    backend.load(src, settings)
    prog, cb = _progress()
    try:
        with prog:
            return score_quant(
                pack, backend, src, settings, groups=groups, quick=quick,
                do_damage=do_damage, warnings=warns, progress=cb,
            )  # fmt: skip
    finally:
        backend.close()


def _get_pack(pack_ref: str | None, src: ResolvedSource, cfg: Settings):
    from quantgap.hub.packs import auto_find_pack, load_pack

    if pack_ref:
        return load_pack(pack_ref)
    return auto_find_pack(src.base_model, cfg)


def _resolve(quant: str, base_url, model_name) -> ResolvedSource:
    from quantgap.hub.resolve import ResolveError, resolve_quant

    try:
        return resolve_quant(quant, base_url=base_url, model_name=model_name)
    except ResolveError as e:
        hint = OLLAMA_HINT if (":" in quant and "/" not in quant and "\\" not in quant) else ""
        raise _fail(f"{e}\n{hint}".strip()) from e


# ---------------------------------------------------------------------------- score


@app.command()
def score(
    quant: Annotated[
        str, typer.Argument(help="Local path, hf://org/repo/file.gguf, or openai://NAME")
    ],
    pack: Annotated[
        str | None,
        typer.Option(help="Pack folder or Hub dataset id (found automatically if omitted)"),
    ] = None,
    backend: Annotated[str, typer.Option(help="auto, llamacpp, mlx, openai or hf")] = "auto",
    base_url: Annotated[
        str | None, typer.Option(help="OpenAI-compatible server, e.g. http://localhost:8000/v1")
    ] = None,
    model_name: Annotated[str | None, typer.Option(help="Model name the server serves")] = None,
    n_gpu_layers: Annotated[
        int | None, typer.Option(help="llama.cpp layers on GPU (-1 = all)")
    ] = None,
    ctx: Annotated[int | None, typer.Option(help="Context size, at least 600")] = None,
    groups: Annotated[
        str | None, typer.Option(help="Comma list: general,chat,code,math,multilingual,json")
    ] = None,
    quick: Annotated[
        bool, typer.Option(help="First 25% of each group: a fast first look, not submittable")
    ] = False,
    json_out: Annotated[
        Path | None, typer.Option("--json", help="Where to save the result JSON")
    ] = None,
    no_damage: Annotated[bool, typer.Option(help="Skip the visible-damage checks")] = False,
) -> None:
    """Score one quant against a pack."""
    from quantgap.report.jsonout import safe_name, write_result
    from quantgap.report.table import print_result
    from quantgap.score.run import ScoreError

    cfg = load_settings()
    if backend not in ("auto", *BACKEND_NAMES):
        raise _fail(f"Unknown backend {backend!r}. Choose auto, {', '.join(BACKEND_NAMES)}.")
    src = _resolve(quant, base_url, model_name)
    try:
        pk = _get_pack(pack, src, cfg)
        res = _score_source(
            src, pk, cfg, backend_name=backend, ctx=ctx, n_gpu_layers=n_gpu_layers,
            base_url=base_url, model_name=model_name,
            groups=[g.strip() for g in groups.split(",")] if groups else None,
            quick=quick, do_damage=not no_damage,
        )  # fmt: skip
    except (ScoreError, RuntimeError, ValueError, FileNotFoundError) as e:
        raise _fail(str(e)) from e
    print_result(console, res)
    out = json_out or Path(cfg.defaults.results_dir) / f"{safe_name(res)}.json"
    write_result(res, out)
    console.print(f"Saved result: {out}")


# ---------------------------------------------------------------------------- compare


@app.command()
def compare(
    quants: Annotated[
        list[str] | None,
        typer.Argument(help="Quants (paths, hf:// URLs) or saved result .json files"),
    ] = None,
    repo: Annotated[
        str | None, typer.Option(help="Hub repo to take quants from, with --glob")
    ] = None,
    glob: Annotated[str, typer.Option(help="File pattern inside --repo")] = "*.gguf",
    pack: Annotated[str | None, typer.Option()] = None,
    backend: Annotated[str, typer.Option()] = "auto",
    budget: Annotated[str | None, typer.Option(help="Memory you have, e.g. 16GB")] = None,
    chart: Annotated[
        Path | None, typer.Option(help="Write the quality-vs-size chart (.svg or .png)")
    ] = None,
    json_out: Annotated[Path | None, typer.Option("--json")] = None,
    ctx: Annotated[int | None, typer.Option()] = None,
    n_gpu_layers: Annotated[int | None, typer.Option()] = None,
    quick: Annotated[bool, typer.Option()] = False,
    no_damage: Annotated[bool, typer.Option()] = False,
) -> None:
    """Score many quants of one model, draw a chart, and say which one fits your memory."""
    from quantgap.hub.resolve import ResolveError, expand_glob
    from quantgap.hub.submit import load_result
    from quantgap.report.chart import draw_chart
    from quantgap.report.jsonout import safe_name, write_report, write_result
    from quantgap.report.table import print_compare
    from quantgap.score.compare import MixedPacksError, build_report
    from quantgap.score.recommend import ASSUMPTION, recommend
    from quantgap.score.run import ScoreError

    cfg = load_settings()
    items = list(quants or [])
    if repo:
        try:
            items += [f"hf://{repo}/{f}" for f in expand_glob(repo, glob)]
        except ResolveError as e:
            raise _fail(str(e)) from e
    if not items:
        raise _fail("Give at least one quant, or --repo ORG/REPO with --glob.")
    budget_bytes = None
    if budget:
        try:
            budget_bytes = parse_size(budget)
        except ValueError as e:
            raise _fail(str(e)) from e

    results: list[ScoreResult] = []
    pk = None
    try:
        for q in items:
            if q.endswith(".json") and Path(q).is_file():
                results.append(load_result(q))
                continue
            src = _resolve(q, None, None)
            if pk is None:
                pk = _get_pack(pack, src, cfg)
            console.print(f"[bold]{src.label}[/bold]")
            res = _score_source(
                src, pk, cfg, backend_name=backend, ctx=ctx, n_gpu_layers=n_gpu_layers,
                base_url=None, model_name=None, groups=None, quick=quick, do_damage=not no_damage,
            )  # fmt: skip
            write_result(res, Path(cfg.defaults.results_dir) / f"{safe_name(res)}.json")
            results.append(res)
        report = build_report(results, budget_bytes)
    except (ScoreError, MixedPacksError, RuntimeError, ValueError, FileNotFoundError) as e:
        raise _fail(str(e)) from e

    print_compare(console, results)
    if budget_bytes:
        rec = recommend(results, budget_bytes)
        console.print(f"\n[bold]{rec.sentence}[/bold]")
        for w in rec.warnings:
            console.print(f"[yellow]{w}[/yellow]")
        console.print(f"[dim]{ASSUMPTION}[/dim]")
    if chart:
        draw_chart(results, chart, budget_bytes=budget_bytes)
        console.print(f"Chart: {chart}")
    if json_out:
        write_report(report, json_out)
        console.print(f"Report: {json_out}")


# ---------------------------------------------------------------------------- pack


@app.command()
def pack(
    model_repo: Annotated[
        str, typer.Argument(help="Reference model on the Hub, e.g. Qwen/Qwen3-8B")
    ],
    revision: Annotated[
        str | None, typer.Option(help="Commit sha (recommended: pins the model)")
    ] = None,
    probes: Annotated[str, typer.Option(help="Probe set version")] = "v1",
    k: Annotated[int, typer.Option(help="Top-k guesses to keep per position")] = 32,
    batch_size: Annotated[int, typer.Option()] = 4,
    device: Annotated[str, typer.Option(help="auto, cuda or cpu")] = "auto",
    out: Annotated[Path | None, typer.Option()] = None,
    push: Annotated[
        str | None, typer.Option(help="Upload to the Hub as ORG/NAME (needs HF_TOKEN)")
    ] = None,
    max_new_tokens: Annotated[int, typer.Option()] = 256,
    license_: Annotated[
        str | None, typer.Option("--license", help="Reference model license, for the dataset card")
    ] = None,
    logp_dtype: Annotated[
        str, typer.Option(help="float32 (default) or float16 (smaller, noisier)")
    ] = "float32",
) -> None:
    """Run the full-size reference model once and save a shareable pack."""
    if (
        importlib.util.find_spec("torch") is None
        or importlib.util.find_spec("transformers") is None
    ):
        raise _fail('Making a pack needs torch and transformers. Run: pip install "quantgap[pack]"')
    from quantgap.hub.packs import push_pack
    from quantgap.pack.runner import build_pack_from_objects, load_reference

    out = out or Path("packs") / model_repo.split("/")[-1]
    console.print(f"Loading {model_repo} ...")
    model, tok, dev, dtype, label = load_reference(model_repo, revision, device)
    if revision is None:
        err.print(
            "[yellow]note:[/yellow] no --revision given. Pin one so the pack can be reproduced."
        )
    prog, cb = _progress()
    with prog:
        manifest = build_pack_from_objects(
            model, tok, out, repo=model_repo, revision=revision, dtype=dtype,
            probes_version=probes, k=k, batch_size=batch_size, max_new_tokens=max_new_tokens,
            logp_dtype=logp_dtype, license_=license_, device=dev, device_label=label,
            progress=lambda d, t: cb("running reference", d, t),
        )  # fmt: skip
    size = sum(f.stat().st_size for f in out.glob("*.parquet")) / 1e6
    console.print(
        f"Pack {manifest.short_name}: {manifest.probes.n_positions:,} positions, {size:.1f} MB -> {out}"
    )
    if push:
        url = push_pack(out, push, os.environ.get("HF_TOKEN"))
        console.print(f"Uploaded: {url}")


@packs_app.command("list")
def packs_list(
    model: Annotated[str | None, typer.Option(help="Filter by model name")] = None,
) -> None:
    """List known packs on the Hub (tag: quantgap-pack)."""
    from quantgap.hub.packs import list_packs

    try:
        found = list_packs(load_settings(), model)
    except Exception as e:
        raise _fail(f"Could not reach the Hub: {e}") from e
    if not found:
        console.print("No packs found. Make one with `quantgap pack MODEL` (docs/packs.md).")
    for ds, m in found:
        console.print(f"{ds}  {m.model.repo}  probes {m.probes.set}  k={m.k}  {m.short_name}")


@packs_app.command("show")
def packs_show(ref: Annotated[str, typer.Argument(help="Pack folder or Hub dataset id")]) -> None:
    """Print a pack's manifest summary."""
    from quantgap.hub.packs import PackNotFound, load_pack
    from quantgap.pack.reader import PackIntegrityError

    try:
        p = load_pack(ref)
    except (PackNotFound, PackIntegrityError, FileNotFoundError) as e:
        raise _fail(str(e)) from e
    m = p.manifest
    console.print(f"[bold]pack:{m.short_name}[/bold]")
    console.print(f"model     {m.model.repo} @ {m.model.revision}  ({m.model.dtype})")
    console.print(
        f"probes    {m.probes.set}, {m.probes.n_sequences} sequences, {m.probes.n_positions:,} positions"
    )
    console.print(f"k         {m.k} ({m.logp_dtype})")
    console.print(f"greedy    {m.greedy.n_prompts} prompts x {m.greedy.max_new_tokens} tokens")
    console.print(f"made on   {m.runner.device}, quantgap {m.runner.quantgap}, {m.created_at}")


# ---------------------------------------------------------------------------- submit / inspect


@app.command()
def submit(
    result: Annotated[Path, typer.Argument(help="Result JSON from `quantgap score`")],
    results_repo: Annotated[str | None, typer.Option(help="Hub dataset to send the PR to")] = None,
    dry_run: Annotated[
        bool, typer.Option(help="Show exactly what would be sent, send nothing")
    ] = False,
) -> None:
    """Open a pull request with your result on the shared results dataset."""
    from quantgap.hub.submit import SubmitError
    from quantgap.hub.submit import submit as do_submit

    repo = results_repo or load_settings().hub.results_repo
    if not dry_run and not os.environ.get("HF_TOKEN"):
        raise _fail(
            "Set HF_TOKEN (a Hugging Face token with write access) to submit. Or use --dry-run."
        )
    try:
        sub, url = do_submit(result, repo, dry_run=dry_run, token=os.environ.get("HF_TOKEN"))
    except (SubmitError, FileNotFoundError, json.JSONDecodeError) as e:
        raise _fail(str(e)) from e
    console.print(f"[bold]File:[/bold] {sub.path}   [bold]Repo:[/bold] {repo}")
    console.print(f"[bold]PR title:[/bold] {sub.pr_title}\n")
    console.print(sub.pr_body)
    console.print("\n[dim](dry run: nothing was sent)[/dim]" if dry_run else f"\nOpened: {url}")


@app.command()
def inspect(result: Annotated[Path, typer.Argument()]) -> None:
    """Pretty-print a saved result."""
    from quantgap.hub.submit import SubmitError, load_result
    from quantgap.report.table import print_result

    try:
        print_result(console, load_result(result))
    except (SubmitError, FileNotFoundError, json.JSONDecodeError) as e:
        raise _fail(str(e)) from e


# ---------------------------------------------------------------------------- doctor


@app.command()
def doctor(
    selftest: Annotated[
        bool, typer.Option(help="Score a tiny built-in model against itself (needs torch)")
    ] = False,
) -> None:
    """Check which backends are installed, whether a GPU is found, and Hugging Face login."""
    console.print(
        f"quantgap {__version__}  python {platform.python_version()}  {platform.system()} {platform.machine()}"
    )
    rows = [
        (
            "llamacpp",
            "llama_cpp",
            'pip install "quantgap[llamacpp]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu',
            "GGUF files",
        ),
        ("mlx", "mlx_lm", 'pip install "quantgap[mlx]"', "Apple silicon (MLX)"),
        ("hf", "transformers", 'pip install "quantgap[hf]"', "Transformers / bitsandbytes"),
        ("pack", "torch", 'pip install "quantgap[pack]"', "making packs"),
    ]
    missing = []
    for name, mod, fix, what in rows:
        ok = importlib.util.find_spec(mod) is not None
        console.print(
            f"  {'[green]ok[/green]     ' if ok else '[yellow]missing[/yellow]'} {name:9} {what}"
            + ("" if ok else f"   -> {escape(fix)}")
        )
        missing += [] if ok else [fix]
    console.print(
        "  [green]ok[/green]      openai    any OpenAI-compatible server (needs --base-url)"
    )
    gpu = "none found"
    if importlib.util.find_spec("torch"):
        import torch

        if torch.cuda.is_available():
            gpu = f"CUDA: {torch.cuda.get_device_name(0)}"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            gpu = "Apple MPS"
    console.print(f"  GPU: {gpu}")
    console.print(
        f"  HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'not set (only needed for pack --push and submit)'}"
    )
    if missing:
        console.print("\nTo score GGUF files on most machines you need only the first line above.")
    if selftest:
        _selftest()


def _selftest() -> None:
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("tokenizers") is None:
        raise _fail('--selftest needs torch and transformers: pip install "quantgap[pack]"')
    import tempfile
    import time

    root = Path(__file__).resolve().parents[2] / "scripts" / "make_tiny_model.py"
    if not root.is_file():
        raise _fail("--selftest needs a source checkout (scripts/make_tiny_model.py).")
    spec = importlib.util.spec_from_file_location("make_tiny_model", root)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from quantgap.backends.hf import HFBackend
    from quantgap.pack import Pack
    from quantgap.pack.runner import build_pack_from_objects
    from quantgap.score.run import score_quant

    t0 = time.time()
    with tempfile.TemporaryDirectory() as d:
        mod.build(Path(d), with_gguf=False)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(Path(d) / "hf")
        model = AutoModelForCausalLM.from_pretrained(Path(d) / "hf")
        probes = Path(__file__).parent / "probes" / "data" / "probes_v1.jsonl"
        build_pack_from_objects(model, tok, Path(d) / "pack", repo="selftest/tiny", dtype="float32",
                                probes_file=probes, k=32, max_new_tokens=16, max_len=200)  # fmt: skip
        b = HFBackend()
        src = ResolvedSource(source="selftest", label="reference", path=Path(d) / "hf", format="hf")
        s = BackendSettings()
        b.load(src, s)
        r = score_quant(Pack.load(Path(d) / "pack"), b, src, s, do_damage=False)
    m = r.metrics
    ok = m.kld_mean < 1e-6 and m.top1_agree == 1.0 and abs(m.ppl_ratio - 1) < 1e-4
    console.print(
        f"  selftest: KLD {m.kld_mean:.2e}, top-1 {m.top1_agree * 100:.1f}%, ppl ratio {m.ppl_ratio:.6f} "
        f"({time.time() - t0:.0f}s) -> {'[green]PASS[/green]' if ok else '[red]FAIL[/red]'}"
    )
    if not ok:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
