"""Typed data models. Pydantic validates JSON files and gives us JSON Schema for free."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PACK_SCHEMA = "quantgap.pack/1"
RESULT_SCHEMA = "quantgap.result/1"

GROUPS = ("general", "chat", "code", "math", "multilingual", "json")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- probes


class ProbeSeq(_Model):
    """One row of the probe set (the texts every model is scored on)."""

    seq_id: int
    group: Literal["general", "chat", "code", "math", "multilingual", "json"]
    kind: Literal["text", "chat"] = "text"
    lang: str = "en"
    text: str | None = None  # kind == "text"
    messages: list[dict[str, str]] | None = None  # kind == "chat"
    license: str
    source: str
    # Used by the damage checks. For text rows the prompt is the first words of the text.
    greedy: bool = False
    greedy_prompt: str | None = None


# --------------------------------------------------------------------------- pack manifest


class ModelRef(_Model):
    repo: str
    revision: str | None = None
    dtype: str = "bfloat16"
    license: str | None = None


class TokenizerInfo(_Model):
    sha256: str | None = None
    vocab_size: int
    bos: bool = True
    bos_id: int | None = None
    eos_id: int | None = None
    check_ids: list[int] = Field(default_factory=list)


class ProbesInfo(_Model):
    set: str
    sha256: str
    n_sequences: int
    n_positions: int


class GreedyInfo(_Model):
    n_prompts: int
    max_new_tokens: int


class RunnerInfo(_Model):
    quantgap: str
    torch: str | None = None
    transformers: str | None = None
    device: str | None = None


class Manifest(_Model):
    schema_: str = Field(default=PACK_SCHEMA, alias="schema")
    model: ModelRef
    tokenizer: TokenizerInfo
    probes: ProbesInfo
    chat_template_sha256: str | None = None
    k: int
    logp_dtype: Literal["float16", "float32"] = "float32"
    greedy: GreedyInfo
    runner: RunnerInfo
    files: dict[str, str] = Field(default_factory=dict)
    created_at: str

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def canonical_json(self) -> str:
        data = self.model_dump(by_alias=True, mode="json")
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    @property
    def pack_id(self) -> str:
        """First 16 hex chars of SHA-256 over the canonical manifest."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:16]

    @property
    def short_name(self) -> str:
        return f"{self.model.repo.split('/')[-1]}@{self.pack_id}"


# --------------------------------------------------------------------------- results


class QuantInfo(_Model):
    source: str
    sha256: str | None = None
    size_bytes: int
    label: str
    publisher: str | None = None
    format: str = "gguf"


class BackendInfo(_Model):
    name: str
    version: str
    settings: dict[str, Any] = Field(default_factory=dict)


class GroupMetrics(_Model):
    kld_mean: float
    top1_agree: float
    n_positions: int = 0


class DamageMetrics(_Model):
    """Every number is stored next to the reference's own number (section 7.4)."""

    n_prompts: int
    loops: int
    loops_ref: int
    loop_rate: float
    wrong_script: int
    wrong_script_ref: int
    wrong_script_rate: float
    json_valid: int
    json_valid_ref: int
    json_n: int
    json_valid_rate: float
    first_divergence_median: float


class Metrics(_Model):
    kld_mean: float
    kld_mean_ci95: tuple[float, float]
    kld_p99: float
    kld_p999: float
    top1_agree: float
    ppl_ratio: float
    n_positions: int
    n_sequences: int
    clamp_count: int = 0
    by_group: dict[str, GroupMetrics] = Field(default_factory=dict)
    damage: DamageMetrics | None = None


class ScoreResult(_Model):
    schema_: str = Field(default=RESULT_SCHEMA, alias="schema")
    pack_id: str
    pack_name: str | None = None
    quant: QuantInfo
    backend: BackendInfo
    metrics: Metrics
    hardware: str
    quantgap: str
    created_at: str
    quick: bool = False
    approximate: bool = False
    nondeterministic: bool = False
    warnings: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def to_json(self) -> str:
        return json.dumps(self.model_dump(by_alias=True, mode="json"), indent=2, ensure_ascii=False)

    @property
    def submittable(self) -> bool:
        return not self.quick and self.quant.sha256 is not None


class CompareReport(_Model):
    pack_id: str
    pack_name: str | None = None
    results: list[ScoreResult]
    budget_bytes: int | None = None
    recommendation: str | None = None
    assumptions: str | None = None


class Submission(_Model):
    """What `quantgap submit` writes into the results dataset."""

    path: str
    pr_title: str
    pr_body: str
    result: ScoreResult
