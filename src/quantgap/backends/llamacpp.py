"""GGUF backend through llama-cpp-python (needs `quantgap[llamacpp]`).

Verified fact (docs/DECISIONS.md, M0): `Llama(..., logits_all=True)` followed by `eval(tokens)`
fills `llm.scores`, an array of shape (n_ctx, n_vocab). Row i holds the logits that come out
after reading token i, so it predicts token i + 1.
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


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self) -> None:
        self.llm: Any = None
        self.settings: BackendSettings | None = None

    def load(self, source: ResolvedSource, settings: BackendSettings) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                'llama-cpp-python is not installed. Run: pip install "quantgap[llamacpp]"'
            ) from e
        if source.path is None or not source.path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {source.path}")
        self.settings = settings
        kwargs: dict[str, Any] = {
            "model_path": str(source.path),
            "n_ctx": settings.ctx,
            "n_gpu_layers": settings.n_gpu_layers,
            "logits_all": True,
            "verbose": False,
        }
        if settings.n_threads:
            kwargs["n_threads"] = settings.n_threads
        self.llm = Llama(**kwargs)

    def tokenizer_fingerprint(self) -> TokFingerprint:
        llm = self.llm
        ids = llm.tokenize(CHECK_STRING.encode("utf-8"), add_bos=False, special=False)
        return TokFingerprint(
            vocab_size=int(llm.n_vocab()),
            bos_id=int(llm.token_bos()) if llm.token_bos() >= 0 else None,
            eos_id=int(llm.token_eos()) if llm.token_eos() >= 0 else None,
            check_ids=[int(i) for i in ids],
        )

    def prompt_logprobs(
        self, token_ids: list[int], positions: np.ndarray, gather_ids: np.ndarray
    ) -> PromptLogprobs:
        llm = self.llm
        llm.reset()
        llm.eval(token_ids)
        positions = np.asarray(positions)
        rows = np.asarray(llm.scores[positions], dtype=np.float32)
        tgt = np.asarray(token_ids, dtype=np.int64)[positions + 1]
        return gather_from_logits(rows, tgt, gather_ids)

    def generate_greedy(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        llm = self.llm
        llm.reset()
        out: list[int] = []
        eos = llm.token_eos()
        for tok in llm.generate(token_ids, top_k=1, top_p=1.0, temp=0.0, reset=True):
            if tok == eos or len(out) >= max_new_tokens:
                break
            out.append(int(tok))
        return out

    def decode(self, ids: list[int]) -> str:
        return self.llm.detokenize(ids).decode("utf-8", errors="replace")

    def version(self) -> str:
        import llama_cpp

        return f"llama-cpp-python {llama_cpp.__version__}"

    def close(self) -> None:
        if self.llm is not None:
            self.llm.close()
            self.llm = None
