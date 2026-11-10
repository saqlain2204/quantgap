# Changelog

## 0.1.0 (unreleased)

First version.

- `quantgap pack`: run a reference model once, save a shareable pack (top-k log probabilities,
  hashes, manifest). Optional upload to the Hugging Face Hub.
- `quantgap score`: KLD (mean, worst 1%, worst 0.1%, bootstrap interval), top-1 agreement,
  perplexity ratio, per-group numbers, and visible-damage checks (loops, wrong script, JSON).
- `quantgap compare`: score many quants, table, quality-vs-size chart, memory-budget recommendation.
- `quantgap submit`: dry-run or open a pull request with a hashed result.
- Backends: llama.cpp (GGUF), MLX (untested), any OpenAI-compatible server with prompt logprobs,
  Transformers.
- Probe set v1: 160 openly licensed texts in 6 groups.
- Static leaderboard builder (`site/build.py`).
