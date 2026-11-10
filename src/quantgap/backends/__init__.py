"""Backend adapters. Each one is imported only when it is asked for, so the default install
(no torch, no llama.cpp) stays light."""

from __future__ import annotations

import platform

from quantgap.backends.base import (
    Backend,
    BackendSettings,
    PromptLogprobs,
    ResolvedSource,
    gather_from_logits,
)

BACKEND_NAMES = ("llamacpp", "mlx", "openai", "hf")

__all__ = [
    "BACKEND_NAMES",
    "Backend",
    "BackendSettings",
    "PromptLogprobs",
    "ResolvedSource",
    "gather_from_logits",
    "get_backend",
    "pick_backend_name",
]


def get_backend(name: str) -> Backend:
    if name == "llamacpp":
        from quantgap.backends.llamacpp import LlamaCppBackend

        return LlamaCppBackend()
    if name == "mlx":
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            raise RuntimeError("The mlx backend only works on Apple silicon Macs.")
        from quantgap.backends.mlx import MLXBackend

        return MLXBackend()
    if name == "openai":
        from quantgap.backends.openai_compat import OpenAIBackend

        return OpenAIBackend()
    if name == "hf":
        from quantgap.backends.hf import HFBackend

        return HFBackend()
    raise ValueError(f"Unknown backend {name!r}. Choose one of: {', '.join(BACKEND_NAMES)}.")


def pick_backend_name(src: ResolvedSource, base_url: str | None) -> str:
    """Guess a backend from what the quant is."""
    if src.format == "openai" or base_url:
        return "openai"
    if src.format == "gguf":
        return "llamacpp"
    if src.format == "mlx":
        return "mlx"
    return "hf"
