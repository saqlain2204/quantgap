"""Any OpenAI-compatible server that returns prompt logprobs (vLLM: `prompt_logprobs`).

This covers NVFP4, AWQ, GPTQ and FP8 quants served by vLLM.

Server facts this adapter relies on (see docs/DECISIONS.md, M0, where they are marked as taken
from the vLLM docs and checked against a fake server, not against a live GPU):
- POST {base}/completions accepts `prompt` as a list of token ids and `prompt_logprobs: N`.
- The reply has `choices[0].prompt_logprobs`: a list with one entry per prompt token. Entry 0 is
  null. Entry i is a dict {"<token id>": {"logprob": float, "rank": int, ...}} holding the actual
  token i plus the server's top-N for that position.
- N is capped by the server (`--max-logprobs`, default 20). We ask for the configured value and
  step down if the server refuses.

A server only tells us its own top-N, not the probability of every reference id. Ids that are
missing come back as NaN and the scorer moves them into the tail bucket (see docs/method.md).
Results from this backend are always marked approximate when that happens.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import numpy as np

from quantgap.backends.base import BackendSettings, PromptLogprobs, ResolvedSource
from quantgap.tokcheck import CHECK_STRING, TokFingerprint

_FALLBACK_N = (20, 10, 5, 1)


class OpenAIBackend:
    name = "openai"

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client
        self.base_url = ""
        self.model_name = ""
        self.n = 20
        self._headers: dict[str, str] = {}

    # ------------------------------------------------------------------ setup
    def load(self, source: ResolvedSource, settings: BackendSettings) -> None:
        if not settings.base_url:
            raise ValueError(
                "The openai backend needs --base-url (for example http://localhost:8000/v1)."
            )
        self.base_url = settings.base_url.rstrip("/")
        self.model_name = settings.model_name or source.extra.get("model_name") or ""
        if not self.model_name:
            raise ValueError("The openai backend needs --model-name (the name the server serves).")
        self.n = settings.max_prompt_logprobs
        key = settings.api_key or os.environ.get("OPENAI_API_KEY")
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}
        if self._client is None:
            self._client = httpx.Client(timeout=600)

    @property
    def _root(self) -> str:
        return self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url

    def _post(self, url: str, payload: dict[str, Any]) -> httpx.Response:
        assert self._client is not None
        return self._client.post(url, json=payload, headers=self._headers)

    # ------------------------------------------------------------------ interface
    def tokenizer_fingerprint(self) -> TokFingerprint:
        ids: list[int] = []
        try:
            r = self._post(
                f"{self._root}/tokenize",
                {"model": self.model_name, "prompt": CHECK_STRING, "add_special_tokens": False},
            )
            if r.status_code == 200:
                ids = [int(t) for t in r.json()["tokens"]]
        except (httpx.HTTPError, KeyError, ValueError):
            pass
        # A server does not tell us its vocabulary size, so -1 means "unknown, skip that check".
        return TokFingerprint(vocab_size=-1, bos_id=None, eos_id=None, check_ids=ids)

    def prompt_logprobs(
        self, token_ids: list[int], positions: np.ndarray, gather_ids: np.ndarray
    ) -> PromptLogprobs:
        entries = self._request_prompt_logprobs(token_ids)
        positions = np.asarray(positions)
        n, k = gather_ids.shape
        gathered = np.full((n, k), np.nan)
        target = np.full(n, np.nan)
        top1 = np.zeros(n, dtype=np.int64)
        for row, p in enumerate(positions):
            entry = entries[p + 1]
            if entry is None:
                raise RuntimeError(f"Server returned no logprobs for token {p + 1}.")
            tid = str(token_ids[p + 1])
            if tid in entry:
                target[row] = entry[tid]["logprob"]
            best = max(entry.items(), key=lambda kv: kv[1]["logprob"])
            top1[row] = int(best[0])
            for j in range(k):
                v = entry.get(str(int(gather_ids[row, j])))
                if v is not None:
                    gathered[row, j] = v["logprob"]
        if np.isnan(target).any():  # defensive: server must always include the actual token
            raise RuntimeError(
                "Server did not return the log probability of the actual next token."
            )
        return PromptLogprobs(gathered, target, top1, approximate=bool(np.isnan(gathered).any()))

    def _request_prompt_logprobs(self, token_ids: list[int]) -> list[dict | None]:
        last = ""
        for n in dict.fromkeys([self.n, *[x for x in _FALLBACK_N if x < self.n]]):
            r = self._post(
                f"{self.base_url}/completions",
                {
                    "model": self.model_name,
                    "prompt": token_ids,
                    "max_tokens": 1,
                    "temperature": 0,
                    "prompt_logprobs": n,
                },
            )
            if r.status_code == 200:
                self.n = n
                data = r.json()["choices"][0].get("prompt_logprobs")
                if data is None:
                    raise RuntimeError(
                        "The server did not return prompt_logprobs. This backend needs vLLM "
                        "or a server with the same extension."
                    )
                return data
            last = r.text[:300]
            if r.status_code != 400:
                break
        raise RuntimeError(f"Server refused the prompt_logprobs request: {last}")

    def generate_greedy(self, token_ids: list[int], max_new_tokens: int) -> list[int]:
        r = self._post(
            f"{self.base_url}/completions",
            {
                "model": self.model_name,
                "prompt": token_ids,
                "max_tokens": max_new_tokens,
                "temperature": 0,
                "logprobs": 1,
                "return_tokens_as_token_ids": True,
            },
        )
        r.raise_for_status()
        toks = r.json()["choices"][0]["logprobs"]["tokens"]
        return [int(t.split(":")[1]) for t in toks]  # entries look like "token_id:123"

    def decode(self, ids: list[int]) -> str:
        r = self._post(f"{self._root}/detokenize", {"model": self.model_name, "tokens": ids})
        r.raise_for_status()
        return r.json()["prompt"]

    def version(self) -> str:
        return f"openai-compatible server at {self.base_url}"

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
