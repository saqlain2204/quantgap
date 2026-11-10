"""Apple silicon backend through mlx-lm (needs `quantgap[mlx]`).

Written from the mlx-lm API (`mlx_lm.load` returns `(model, tokenizer)`; calling the model on an
array of token ids with shape (1, T) returns logits of shape (1, T, V)). It has NOT been run on a
Mac by the author yet: see docs/DECISIONS.md. Run `quantgap doctor` on your Mac and open an issue
if anything fails.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from quantgap.backends.base import (
    BackendSettings,
    PromptLogprobs,
    ResolvedSource,
    gather_from_logits,
)
from quantgap.tokcheck import CHECK_STRING, TokFingerprint


class MLXBackend:
    name = "mlx"

    def __init__(self) -> None:
        self.model: Any = None
        self.tok: Any = None

    def load(self, source: ResolvedSource, settings: BackendSettings) -> None:
        try:
            from mlx_lm import load
        except ImportError as e:  # pragma: no cover
            raise RuntimeError('mlx-lm is not installed. Run: pip install "quantgap[mlx]"') from e
        where = str(source.path) if source.path else source.source.removeprefix("hf://")
        self.model, self.tok = load(where)

    def _hf_tok(self) -> Any:
        return getattr(self.tok, "_tokenizer", self.tok)

    def tokenizer_fingerprint(self) -> TokFingerprint:
        t = self._hf_tok()
        ids = list(t(CHECK_STRING, add_special_tokens=False)["input_ids"])
        vocab = len(t)
        try:  # the real embedding size is what matters for padding ids
            vocab = int(self.model.model.embed_tokens.weight.shape[0])
        except Exception:
            pass
        return TokFingerprint(vocab, t.bos_token_id, t.eos_token_id, ids)

    def prompt_logprobs(
        self, token_ids: list[int], positions: np.ndarray, gather_ids: np.ndarray
    ) -> PromptLogprobs:
        import mlx.core as mx

        logits = self.model(mx.array([token_ids]))[0]
        rows = np.array(logits.astype(mx.float32))[np.asarray(positions)]
        tgt = np.asarray(token_ids, dtype=np.int64)[np.asarray(positions) + 1]
        return gather_from_logits(rows, tgt, gather_ids)

    def generate_greedy(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        import mlx.core as mx

        ids = list(token_ids)
        out: list[int] = []
        eos = self.tok.eos_token_id
        for _ in range(max_new_tokens):  # no KV cache: slow but simple and exact
            logits = self.model(mx.array([ids]))[0, -1]
            nxt = int(mx.argmax(logits).item())
            if nxt == eos:
                break
            out.append(nxt)
            ids.append(nxt)
        return out

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def version(self) -> str:
        import mlx_lm

        return f"mlx-lm {getattr(mlx_lm, '__version__', 'unknown')}"

    def close(self) -> None:
        self.model = None
        self.tok = None
