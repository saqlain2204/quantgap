"""Transformers backend (needs `quantgap[hf]`). Also the reference for self-tests."""

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


class HFBackend:
    name = "hf"

    def __init__(self) -> None:
        self.model: Any = None
        self.tok: Any = None
        self.device: Any = None

    def load(self, source: ResolvedSource, settings: BackendSettings) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        where = str(source.path) if source.path else source.source.removeprefix("hf://")
        self.tok = AutoTokenizer.from_pretrained(where)
        kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        self.model = AutoModelForCausalLM.from_pretrained(where, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def tokenizer_fingerprint(self) -> TokFingerprint:
        return TokFingerprint(
            vocab_size=int(self.model.get_output_embeddings().weight.shape[0])
            if self.model.get_output_embeddings() is not None
            else len(self.tok),
            bos_id=self.tok.bos_token_id,
            eos_id=self.tok.eos_token_id,
            check_ids=list(self.tok(CHECK_STRING, add_special_tokens=False)["input_ids"]),
        )

    def prompt_logprobs(
        self, token_ids: list[int], positions: np.ndarray, gather_ids: np.ndarray
    ) -> PromptLogprobs:
        import torch

        ids = torch.tensor([token_ids], device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids=ids).logits[0]
        rows = logits[torch.from_numpy(np.asarray(positions)).to(logits.device)]
        rows = rows.float().cpu().numpy()
        tgt = np.asarray(token_ids, dtype=np.int64)[np.asarray(positions) + 1]
        return gather_from_logits(rows, tgt, gather_ids)

    def generate_greedy(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        import torch

        inp = torch.tensor([token_ids], device=self.device)
        eos = self.tok.eos_token_id
        with torch.no_grad():
            out = self.model.generate(
                input_ids=inp,
                attention_mask=torch.ones_like(inp),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tok.pad_token_id if self.tok.pad_token_id is not None else eos,
            )
        new = out[0, len(token_ids) :].tolist()
        if eos is not None and eos in new:
            new = new[: new.index(eos)]
        return new

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def version(self) -> str:
        import transformers

        return f"transformers {transformers.__version__}"

    def close(self) -> None:
        self.model = None
        self.tok = None
