"""The small interface every backend adapter has to provide."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from quantgap.score.kld import log_softmax
from quantgap.tokcheck import TokFingerprint

LOGSOFTMAX_CHUNK = 64  # positions per float64 log-softmax, keeps memory flat for big vocabularies


@dataclass
class ResolvedSource:
    """A quant that has been located on disk (or on a server)."""

    source: str  # what the user typed, or the canonical hf:// URL
    label: str  # short name, for example "UD-Q4_K_XL"
    path: Path | None = None  # local file or folder; None for a server
    size_bytes: int = 0
    format: str = "gguf"
    publisher: str | None = None
    base_model: str | None = None
    sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendSettings:
    ctx: int = 2048
    n_gpu_layers: int = -1
    n_threads: int | None = None
    base_url: str | None = None
    model_name: str | None = None
    max_prompt_logprobs: int = 20
    api_key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        """Settings that go into the result file (never the API key)."""
        d = {"ctx": self.ctx}
        if self.n_gpu_layers != -1:
            d["n_gpu_layers"] = self.n_gpu_layers
        else:
            d["n_gpu_layers"] = -1
        if self.n_threads:
            d["n_threads"] = self.n_threads
        if self.base_url:
            d["base_url"] = self.base_url
        if self.model_name:
            d["model_name"] = self.model_name
        d.update(self.extra)
        return d


@dataclass
class PromptLogprobs:
    """What a backend returns for one sequence, at the requested positions only."""

    gathered: np.ndarray  # (n, k) log prob of each requested id; NaN = could not tell
    target: np.ndarray  # (n,) log prob of the actual next token
    top1: np.ndarray  # (n,) the model's own best next token
    approximate: bool = False  # True for servers that only return their own top-N


class Backend(Protocol):
    name: str

    def load(self, source: ResolvedSource, settings: BackendSettings) -> None: ...

    def tokenizer_fingerprint(self) -> TokFingerprint: ...

    def prompt_logprobs(
        self, token_ids: list[int], positions: np.ndarray, gather_ids: np.ndarray
    ) -> PromptLogprobs:
        """Teacher forcing. Feed `token_ids`, and at each position p in `positions` return the
        model's log probability for every id in gather_ids[row], plus for token p + 1."""
        ...

    def generate_greedy(self, token_ids: list[int], max_new_tokens: int) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    def version(self) -> str: ...

    def close(self) -> None: ...


def gather_from_logits(
    logits: np.ndarray, target_ids: np.ndarray, gather_ids: np.ndarray
) -> PromptLogprobs:
    """Shared maths for backends that hand back full logits.

    logits: (n, V) raw scores, one row per scored position.
    target_ids: (n,) the actual next token at each row.
    gather_ids: (n, k) the reference's top-k ids at each row.
    Uses a stable float64 log-softmax, in chunks so big vocabularies stay cheap.
    """
    n, k = gather_ids.shape
    gathered = np.empty((n, k), dtype=np.float64)
    target = np.empty(n, dtype=np.float64)
    top1 = np.empty(n, dtype=np.int64)
    vocab = logits.shape[1]
    for a in range(0, n, LOGSOFTMAX_CHUNK):
        b = min(n, a + LOGSOFTMAX_CHUNK)
        lp = log_softmax(logits[a:b])
        ids = gather_ids[a:b]
        oob = ids >= vocab  # reference ids beyond this model's vocabulary
        g = np.take_along_axis(lp, np.where(oob, 0, ids), axis=1)
        gathered[a:b] = np.where(oob, -np.inf, g)
        tid = target_ids[a:b]
        target[a:b] = lp[np.arange(b - a), np.minimum(tid, vocab - 1)]
        top1[a:b] = lp.argmax(axis=1)
    return PromptLogprobs(gathered, target, top1, approximate=False)
