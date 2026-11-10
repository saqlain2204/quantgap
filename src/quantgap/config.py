"""Optional `quantgap.toml` (current folder, then ~/.config/quantgap/) plus environment variables."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class Defaults(BaseModel):
    model_config = ConfigDict(extra="ignore")
    backend: str = "auto"
    ctx: int = 2048
    results_dir: str = "./quantgap-results"


class LlamaCppCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    n_gpu_layers: int = -1
    n_threads: int | None = None


class OpenAICfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    base_url: str = "http://localhost:8000/v1"
    model_name: str | None = None
    max_prompt_logprobs: int = 20


class HubCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    results_repo: str = "quantgap/results"
    packs_org: str = "quantgap-packs"


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    defaults: Defaults = Defaults()
    llamacpp: LlamaCppCfg = LlamaCppCfg()
    openai: OpenAICfg = OpenAICfg()
    hub: HubCfg = HubCfg()


def _candidate_paths() -> list[Path]:
    return [Path.cwd() / "quantgap.toml", Path.home() / ".config" / "quantgap" / "quantgap.toml"]


def load_settings() -> Settings:
    data: dict = {}
    for p in _candidate_paths():
        if p.is_file():
            with open(p, "rb") as f:
                data = tomllib.load(f)
            break
    s = Settings.model_validate(data)
    if env := os.environ.get("QUANTGAP_RESULTS_REPO"):
        s.hub.results_repo = env
    return s
