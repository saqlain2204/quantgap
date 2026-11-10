"""Score one quant against one pack. Backend-agnostic: everything backend-specific is behind the
`Backend` interface."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from quantgap import __version__
from quantgap.backends.base import Backend, BackendSettings, ResolvedSource
from quantgap.models import (
    BackendInfo,
    DamageMetrics,
    GroupMetrics,
    Metrics,
    QuantInfo,
    ScoreResult,
)
from quantgap.pack.reader import GreedyRow, Pack
from quantgap.score import damage as dmg
from quantgap.score.kld import kld_topk, ppl_ratio
from quantgap.score.stats import bootstrap_mean_ci, percentile
from quantgap.tokcheck import TokenizerMismatch, check_tokenizer
from quantgap.util import hardware_string, jsonable_float, utc_now

QUICK_FRACTION = 0.25
QUICK_GREEDY_PER_GROUP = 2
NONDETERMINISM_TOL = 1e-3

Progress = Callable[[str, int, int], None]


class ScoreError(RuntimeError):
    """A clear, user-facing reason scoring cannot run."""


@dataclass
class _SeqScore:
    group: str
    kl: np.ndarray
    agree: np.ndarray
    lq_target: np.ndarray
    lp_target: np.ndarray


def select_sequences(
    pack: Pack, groups: list[str] | None, quick: bool
) -> list[int]:
    """Sequence ids to score. --groups filters; --quick keeps the first 25% of each group."""
    rows = pack.probes()
    by_group: dict[str, list[int]] = {}
    for r in rows:
        by_group.setdefault(r.group, []).append(r.seq_id)
    chosen: list[int] = []
    for g, ids in by_group.items():
        if groups and g not in groups:
            continue
        ids = sorted(ids)
        if quick:
            ids = ids[: max(1, math.ceil(len(ids) * QUICK_FRACTION))]
        chosen += ids
    if not chosen:
        raise ScoreError(f"No sequences match --groups {groups}. Groups in this pack: {sorted(by_group)}.")
    return chosen


def check_context(pack: Pack, settings: BackendSettings, seq_ids: list[int]) -> None:
    longest = max(len(r.token_ids) for r in pack.probes() if r.seq_id in set(seq_ids))
    if settings.ctx < longest + 8:
        raise ScoreError(
            f"--ctx {settings.ctx} is too small. The longest probe has {longest} tokens, "
            f"so use --ctx {longest + 8} or more (we suggest 2048)."
        )


def run_damage(
    backend: Backend, pack: Pack, quick: bool, progress: Progress | None
) -> DamageMetrics | None:
    rows: list[GreedyRow] = pack.greedy()
    if quick:
        seen: dict[str, int] = {}
        keep = []
        for r in rows:
            seen[r.group] = seen.get(r.group, 0) + 1
            if seen[r.group] <= QUICK_GREEDY_PER_GROUP:
                keep.append(r)
        rows = keep
    max_new = pack.manifest.greedy.max_new_tokens
    loops = loops_ref = ws = ws_ref = jv = jv_ref = jn = 0
    divs: list[int] = []
    try:
        for i, r in enumerate(rows):
            if progress:
                progress("generating", i, len(rows))
            q_ids = backend.generate_greedy(list(r.prompt_ids), max_new)
            q_text = backend.decode(q_ids)
            divs.append(dmg.first_divergence(q_ids, r.ref_ids))
            loops += dmg.is_loop(q_ids)
            loops_ref += dmg.is_loop(r.ref_ids)
            if r.group in dmg.SCRIPT_CHECK_GROUPS:
                ws += dmg.has_wrong_script(q_text, r.ref_text)
                # The reference is compared with itself, so it flags nothing by definition.
            if r.group == "json":
                jn += 1
                jv += dmg.json_valid(q_text)
                jv_ref += dmg.json_valid(r.ref_text)
    except Exception as e:  # a backend that cannot generate should not lose the KLD result
        raise ScoreError(f"Visible-damage checks failed: {e}. Use --no-damage to skip them.") from e
    n = max(1, len(rows))
    return DamageMetrics(
        n_prompts=len(rows),
        loops=int(loops),
        loops_ref=int(loops_ref),
        loop_rate=loops / n,
        wrong_script=int(ws),
        wrong_script_ref=int(ws_ref),
        wrong_script_rate=ws / n,
        json_valid=int(jv),
        json_valid_ref=int(jv_ref),
        json_n=jn,
        json_valid_rate=(jv / jn) if jn else 0.0,
        first_divergence_median=dmg.median_or_zero(divs),
    )


def score_quant(
    pack: Pack,
    backend: Backend,
    source: ResolvedSource,
    settings: BackendSettings,
    *,
    groups: list[str] | None = None,
    quick: bool = False,
    do_damage: bool = True,
    warnings: list[str] | None = None,
    progress: Progress | None = None,
) -> ScoreResult:
    warnings = list(warnings or [])
    manifest = pack.manifest
    topk = pack.topk()
    probes = {r.seq_id: r for r in pack.probes()}
    seq_ids = select_sequences(pack, groups, quick)
    check_context(pack, settings, seq_ids)

    # Tokenizer check (never silently fixed).
    fp = backend.tokenizer_fingerprint()
    max_id = max(int(t.ids.max()) for t in topk.values() if len(t.ids))
    try:
        check_tokenizer(manifest.tokenizer, fp, max_id)
    except TokenizerMismatch as e:
        raise ScoreError(str(e)) from e

    # E6: run the first sequence twice and see if the numbers move.
    nondeterministic = False
    first = next(s for s in seq_ids if len(topk[s].pos))
    a = backend.prompt_logprobs(list(probes[first].token_ids), topk[first].pos, topk[first].ids)
    b = backend.prompt_logprobs(list(probes[first].token_ids), topk[first].pos, topk[first].ids)
    both = ~np.isnan(a.gathered) & ~np.isnan(b.gathered)
    if both.any() and np.max(np.abs(a.gathered[both] - b.gathered[both])) > NONDETERMINISM_TOL:
        nondeterministic = True
        warnings.append("Running the same text twice gave different numbers (non-deterministic).")

    scores: list[_SeqScore] = []
    approximate = False
    clamps = 0
    for i, sid in enumerate(seq_ids):
        if progress:
            progress("scoring", i, len(seq_ids))
        t = topk[sid]
        if len(t.pos) == 0:
            continue
        pl = backend.prompt_logprobs(list(probes[sid].token_ids), t.pos, t.ids)
        approximate |= pl.approximate
        res = kld_topk(t.logp, pl.gathered)
        clamps += res.clamps
        scores.append(
            _SeqScore(
                probes[sid].group,
                res.kl,
                pl.top1 == t.ids[:, 0],
                pl.target,
                t.logp_target,
            )
        )
    if progress:
        progress("scoring", len(seq_ids), len(seq_ids))

    kl = np.concatenate([s.kl for s in scores])
    agree = np.concatenate([s.agree for s in scores])
    lq = np.concatenate([s.lq_target for s in scores])
    lp = np.concatenate([s.lp_target for s in scores])
    sums = np.array([s.kl.sum() for s in scores])
    counts = np.array([len(s.kl) for s in scores])
    ci = bootstrap_mean_ci(sums, counts)

    by_group: dict[str, GroupMetrics] = {}
    for g in dict.fromkeys(s.group for s in scores):
        sel = [s for s in scores if s.group == g]
        gk = np.concatenate([s.kl for s in sel])
        ga = np.concatenate([s.agree for s in sel])
        by_group[g] = GroupMetrics(
            kld_mean=jsonable_float(float(gk.mean())),
            top1_agree=jsonable_float(float(ga.mean())),
            n_positions=len(gk),
        )

    if clamps > max(10, 0.001 * len(kl)):
        warnings.append(
            f"{clamps} probabilities were raised to 1e-12 before taking logs. A high number "
            "means something is broken (wrong tokenizer or a damaged model)."
        )
    if approximate:
        warnings.append(
            "approximate (server top-N): the server returned only its own top-N, so some "
            "reference ids were counted in the tail bucket."
        )

    damage_metrics = None
    if do_damage:
        damage_metrics = run_damage(backend, pack, quick, progress)

    metrics = Metrics(
        kld_mean=jsonable_float(float(kl.mean())),
        kld_mean_ci95=(jsonable_float(ci[0]), jsonable_float(ci[1])),
        kld_p99=jsonable_float(percentile(kl, 99)),
        kld_p999=jsonable_float(percentile(kl, 99.9)),
        top1_agree=jsonable_float(float(agree.mean())),
        ppl_ratio=jsonable_float(ppl_ratio(lq, lp)),
        n_positions=len(kl),
        n_sequences=len(scores),
        clamp_count=clamps,
        by_group=by_group,
        damage=damage_metrics,
    )
    return ScoreResult(
        pack_id=pack.pack_id,
        pack_name=manifest.short_name,
        quant=QuantInfo(
            source=source.source,
            sha256=source.sha256,
            size_bytes=source.size_bytes,
            label=source.label,
            publisher=source.publisher,
            format=source.format,
        ),
        backend=BackendInfo(name=backend.name, version=backend.version(), settings=settings.public()),
        metrics=metrics,
        hardware=hardware_string(),
        quantgap=__version__,
        created_at=utc_now(),
        quick=quick,
        approximate=approximate,
        nondeterministic=nondeterministic,
        warnings=warnings,
    )
