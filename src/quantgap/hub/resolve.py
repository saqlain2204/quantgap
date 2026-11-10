"""Turn what the user typed into a quant on disk: local paths, hf:// URLs, repo globs."""

from __future__ import annotations

import fnmatch
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantgap.backends.base import ResolvedSource
from quantgap.util import free_disk_bytes, human_size, sha256_file

_LABEL_RE = re.compile(
    r"((?:UD-)?(?:IQ\d(?:_[A-Z0-9]+)*|Q\d(?:_[A-Z0-9]+)*|MXFP4(?:_MOE)?|BF16|F16|F32|NVFP4))",
    re.IGNORECASE,
)


class ResolveError(ValueError):
    """The quant reference could not be understood or found. The message says what to try."""


@dataclass
class HFRef:
    repo: str
    path: str | None = None


def parse_hf_url(url: str) -> HFRef:
    """`hf://org/repo` or `hf://org/repo/sub/folder/file.gguf`."""
    if not url.startswith("hf://"):
        raise ResolveError(f"{url!r} is not an hf:// URL. Example: hf://org/repo/file.gguf")
    parts = [p for p in url[len("hf://") :].split("/") if p]
    if len(parts) < 2:
        raise ResolveError(
            f"{url!r} needs at least org/repo. Example: hf://unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
        )
    return HFRef(repo="/".join(parts[:2]), path="/".join(parts[2:]) or None)


def label_from_filename(name: str) -> str:
    stem = Path(name).stem
    m = _LABEL_RE.search(stem)
    return m.group(1) if m else stem


def expand_glob(repo: str, pattern: str, files: list[str] | None = None) -> list[str]:
    """Repo file paths matching a glob. `files` can be given (tests); otherwise asks the Hub."""
    if files is None:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo)
    hits = sorted(
        f for f in files if fnmatch.fnmatch(f, pattern) or fnmatch.fnmatch(Path(f).name, pattern)
    )
    # Split GGUFs ("-00001-of-00003.gguf") are not supported yet: keep only single files or first parts.
    out = []
    for f in hits:
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", f)
        if m and m.group(1) != "00001":
            continue
        out.append(f)
    if not out:
        raise ResolveError(f"No files in {repo} match {pattern!r}.")
    return out


# ------------------------------------------------------------------ GGUF metadata (no deps)

_GGUF_TYPES = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


def read_gguf_metadata(path: str | Path, wanted: set[str] | None = None) -> dict[str, Any]:
    """Read key/value metadata from a GGUF header. Arrays are skipped. Never loads tensors."""
    out: dict[str, Any] = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ResolveError(f"{path} is not a GGUF file.")
        (_version,) = struct.unpack("<I", f.read(4))
        _n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

        def read_str() -> str:
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", errors="replace")

        def read_value(t: int) -> Any:
            if t == 8:
                return read_str()
            if t == 9:  # array
                (et,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                for _ in range(n):
                    read_value(et)
                return None
            fmt = _GGUF_TYPES[t]
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        for _ in range(n_kv):
            key = read_str()
            (t,) = struct.unpack("<I", f.read(4))
            val = read_value(t)
            if val is not None and (wanted is None or key in wanted):
                out[key] = val
            if wanted is not None and wanted <= out.keys():
                break
    return out


def gguf_base_model(path: str | Path) -> str | None:
    """The `org/repo` of the model this GGUF was made from, when the file says so."""
    keys = {"general.base_model.0.repo_url", "general.base_model.0.name"}
    try:
        meta = read_gguf_metadata(path, keys)
    except (OSError, ResolveError, struct.error, KeyError):
        return None
    url = meta.get("general.base_model.0.repo_url", "")
    m = re.search(r"huggingface\.co/([^/]+/[^/?#]+)", url)
    if m:
        return m.group(1)
    return meta.get("general.base_model.0.name")


# ------------------------------------------------------------------ resolve


def _dir_format(path: Path) -> str:
    cfg = path / "config.json"
    if cfg.is_file() and '"quantization"' in cfg.read_text(encoding="utf-8", errors="ignore"):
        return "mlx"
    return "hf"


def _check_disk(size: int, cache_dir: str | None) -> None:
    from huggingface_hub import constants

    where = cache_dir or constants.HF_HUB_CACHE
    free = free_disk_bytes(where)
    need = int(size * 1.1)
    if free < need:
        raise ResolveError(
            f"Not enough disk space in {where}: need about {human_size(need)}, "
            f"{human_size(free)} free. Set HF_HOME to a bigger drive."
        )
    print(f"Downloading to {where} ({human_size(size)}, {human_size(free)} free)")


def resolve_quant(
    quant: str,
    *,
    base_url: str | None = None,
    model_name: str | None = None,
    compute_hash: bool = True,
) -> ResolvedSource:
    """Find a quant. Downloads from the Hub when given an hf:// URL."""
    if quant.startswith("openai://"):
        if not base_url:
            raise ResolveError("openai:// needs --base-url, for example http://localhost:8000/v1")
        name = quant[len("openai://") :] or (model_name or "")
        return ResolvedSource(
            source=quant,
            label=name or "server-model",
            format="openai",
            extra={"model_name": model_name or name},
        )
    if quant.startswith("hf://"):
        ref = parse_hf_url(quant)
        return _resolve_hf(ref, quant)
    p = Path(quant).expanduser()
    if p.is_file():
        src = ResolvedSource(
            source=str(p),
            label=label_from_filename(p.name) if p.suffix == ".gguf" else p.stem,
            path=p,
            size_bytes=p.stat().st_size,
            format="gguf" if p.suffix.lower() == ".gguf" else "hf",
        )
        if src.format == "gguf":
            src.base_model = gguf_base_model(p)
        if compute_hash:
            src.sha256 = sha256_file(p)
        return src
    if p.is_dir():
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
        return ResolvedSource(
            source=str(p), label=p.name, path=p, size_bytes=size, format=_dir_format(p)
        )
    raise ResolveError(
        f"Cannot find {quant!r}. Use a local file or folder, hf://org/repo/file.gguf, "
        "or openai://NAME with --base-url."
    )


def _resolve_hf(ref: HFRef, original: str) -> ResolvedSource:
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi()
    if ref.path and ref.path.endswith(".gguf"):
        info = api.get_paths_info(ref.repo, [ref.path])
        if not info:
            raise ResolveError(f"{ref.path} was not found in {ref.repo}.")
        size = info[0].size or 0
        sha = info[0].lfs.sha256 if getattr(info[0], "lfs", None) else None
        _check_disk(size, None)
        local = Path(hf_hub_download(ref.repo, ref.path))
        if sha is None:
            sha = sha256_file(local)
        return ResolvedSource(
            source=f"hf://{ref.repo}/{ref.path}",
            label=label_from_filename(ref.path),
            path=local,
            size_bytes=size or local.stat().st_size,
            format="gguf",
            publisher=ref.repo.split("/")[0],
            base_model=gguf_base_model(local),
            sha256=sha,
        )
    if ref.path:
        raise ResolveError(
            f"{original!r}: only .gguf files or whole repos (for MLX or Transformers) are supported."
        )
    # whole repo: MLX or Transformers folder
    files = api.model_info(ref.repo, files_metadata=True).siblings or []
    size = sum(s.size or 0 for s in files if s.rfilename.endswith((".safetensors", ".bin")))
    _check_disk(size, None)
    local = Path(snapshot_download(ref.repo))
    return ResolvedSource(
        source=f"hf://{ref.repo}",
        label=ref.repo.split("/")[-1],
        path=local,
        size_bytes=size,
        format=_dir_format(local),
        publisher=ref.repo.split("/")[0],
    )
