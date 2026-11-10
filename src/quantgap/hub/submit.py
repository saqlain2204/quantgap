"""Send a result to the shared results dataset as a pull request."""

from __future__ import annotations

import json
from pathlib import Path

from quantgap.models import RESULT_SCHEMA, ScoreResult, Submission


class SubmitError(ValueError):
    pass


def load_result(path: str | Path) -> ScoreResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != RESULT_SCHEMA:
        raise SubmitError(f"{path} is not a quantgap result (schema {data.get('schema')!r}).")
    return ScoreResult.model_validate(data)


def build_submission(result: ScoreResult) -> Submission:
    """Check a result is submittable and describe exactly what would be sent."""
    if result.quick:
        raise SubmitError(
            "This is a --quick result. Run without --quick to get a submittable result."
        )
    if not result.quant.sha256:
        raise SubmitError(
            "This result has no file hash, so nobody could re-check it. Score a GGUF file or a "
            "downloaded model folder, not a bare server."
        )
    path = f"results/{result.pack_id}/{result.quant.sha256}.json"
    m = result.metrics
    d = m.damage
    lines = [
        f"Result for **{result.quant.label}** ({result.quant.source})",
        "",
        f"- pack: `{result.pack_name or result.pack_id}` (id `{result.pack_id}`)",
        f"- quant file SHA-256: `{result.quant.sha256}`",
        f"- backend: {result.backend.name} ({result.backend.version})",
        f"- hardware: {result.hardware}",
        f"- KLD mean: {m.kld_mean:.5f} (95% CI {m.kld_mean_ci95[0]:.5f} to {m.kld_mean_ci95[1]:.5f})",
        f"- top-1 agreement: {m.top1_agree * 100:.2f}%",
        f"- approximate (server top-N): {'yes' if result.approximate else 'no'}",
    ]
    if d:
        lines.append(
            f"- loops: {d.loops} (reference {d.loops_ref}), wrong script: {d.wrong_script}, "
            f"valid JSON: {d.json_valid}/{d.json_n} (reference {d.json_valid_ref})"
        )
    lines += [
        "",
        "To re-check: `quantgap score <same file> --pack <same pack>` should give a KLD mean within 2%.",
        "",
        "Submitted with quantgap " + result.quantgap + ".",
    ]
    return Submission(
        path=path,
        pr_title=f"Add result: {result.quant.label} on {result.pack_name or result.pack_id}",
        pr_body="\n".join(lines),
        result=result,
    )


def submit(
    result_file: str | Path,
    results_repo: str,
    dry_run: bool = False,
    token: str | None = None,
) -> tuple[Submission, str | None]:
    """Returns (submission, PR url). With dry_run nothing is sent and the URL is None."""
    sub = build_submission(load_result(result_file))
    if dry_run:
        return sub, None
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    info = api.create_commit(
        repo_id=results_repo,
        repo_type="dataset",
        operations=[CommitOperationAdd(sub.path, sub.result.to_json().encode("utf-8"))],
        commit_message=sub.pr_title,
        commit_description=sub.pr_body,
        create_pr=True,
    )
    return sub, info.pr_url
