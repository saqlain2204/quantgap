"""Find, download and publish packs (Hugging Face datasets tagged `quantgap-pack`)."""

from __future__ import annotations

import json
from pathlib import Path

from quantgap.config import Settings
from quantgap.models import Manifest
from quantgap.pack.reader import Pack

PACK_TAG = "quantgap-pack"


class PackNotFound(RuntimeError):
    pass


def _norm(s: str | None) -> str:
    return (s or "").strip().lower().removeprefix("https://huggingface.co/")


def list_packs(settings: Settings, model_query: str | None = None) -> list[tuple[str, Manifest]]:
    """(dataset id, manifest) for every known pack, optionally filtered by model name."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    out: list[tuple[str, Manifest]] = []
    for ds in api.list_datasets(filter=PACK_TAG, author=settings.hub.packs_org):
        if model_query and model_query.lower() not in ds.id.lower():
            continue
        try:
            path = hf_hub_download(ds.id, "manifest.json", repo_type="dataset")
            out.append((ds.id, Manifest.model_validate(json.loads(Path(path).read_text("utf-8")))))
        except Exception:
            continue
    return out


def load_pack(ref: str, verify: bool = True) -> Pack:
    """`ref` is a local folder or a Hub dataset id like `quantgap-packs/Qwen3-8B`."""
    p = Path(ref).expanduser()
    if p.is_dir():
        return Pack.load(p, verify=verify)
    from huggingface_hub import snapshot_download

    try:
        local = snapshot_download(ref, repo_type="dataset")
    except Exception as e:
        raise PackNotFound(
            f"Could not find a pack at {ref!r} (not a folder, not a Hub dataset): {e}"
        ) from e
    return Pack.load(local, verify=verify)


def auto_find_pack(base_model: str | None, settings: Settings) -> Pack:
    """Find the pack that matches a quant's base model, or explain what to do."""
    if not base_model:
        raise PackNotFound(
            "This quant does not say which model it was made from, so no pack can be chosen "
            "automatically. Pass --pack PACK_REF (a pack folder or a Hub dataset id). "
            "List packs with: quantgap packs list"
        )
    matches = [
        (ds, m) for ds, m in list_packs(settings) if _norm(m.model.repo) == _norm(base_model)
    ]
    if not matches:
        raise PackNotFound(
            f"No pack found for {base_model}. Make one with: quantgap pack {base_model} "
            "(see docs/packs.md), or pass --pack."
        )
    if len(matches) > 1:
        options = "\n".join(f"  {ds}  ({m.short_name}, probes {m.probes.set})" for ds, m in matches)
        raise PackNotFound(
            f"More than one pack matches {base_model}. Pick one with --pack:\n{options}"
        )
    return load_pack(matches[0][0])


def fine_tune_warning(base_model: str | None, pack: Pack) -> str | None:
    """E2: the quant says it comes from a different model than the pack's."""
    if base_model and _norm(base_model) != _norm(pack.manifest.model.repo):
        return (
            f"This looks like a fine-tune ({base_model}, not {pack.manifest.model.repo}). "
            "Its drift includes the fine-tuning, not only quantization."
        )
    return None


def dataset_card(m: Manifest) -> str:
    lic = m.model.license or "unknown (check the model page before sharing)"
    return f"""---
license: cc-by-4.0
tags:
- {PACK_TAG}
- quantization
pretty_name: quantgap pack for {m.model.repo}
---

# quantgap pack: {m.model.repo}

Pack id: `{m.pack_id}`. Made with quantgap {m.runner.quantgap}.

This pack holds the reference model's top-{m.k} next-token log probabilities on the quantgap
probe set `{m.probes.set}` ({m.probes.n_positions:,} positions). Use it with
`quantgap score QUANT --pack <this dataset>` to measure how far a quantized copy of the model
drifts from the original, without running the original.

- Reference model: `{m.model.repo}` @ `{m.model.revision}` ({m.model.dtype})
- Reference model license: **{lic}**
- The pack contains model outputs (log probabilities) on openly licensed texts. It does not
  contain the model weights.
"""


def push_pack(pack_dir: str | Path, repo_id: str, token: str | None = None) -> str:
    from huggingface_hub import HfApi

    pack = Pack.load(pack_dir)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    (Path(pack_dir) / "README.md").write_text(dataset_card(pack.manifest), encoding="utf-8")
    api.upload_folder(folder_path=str(pack_dir), repo_id=repo_id, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo_id}"
