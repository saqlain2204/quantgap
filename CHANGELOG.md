# Changelog

## Unreleased

- 15 full-mode sample results for Qwen2.5-0.5B-Instruct (two publishers), measured on a free Colab T4
  GPU, replace the 7 quick-mode ones. They are submittable and each carries its file hash.
- Cross-machine check: the same quant on the same subset gave KLD 0.002720 (laptop CPU) and 0.002694
  (Colab GPU), a 0.95% difference.
- The 15 results are in the shared results dataset and the leaderboard site is live with them. Chart labels on the site
  now show the publisher when two files share a label, and avoid overlapping.
- New notebook `notebooks/score_quants_colab.ipynb` for scoring quants on a free Colab GPU.

## 0.1.1

- README works on PyPI: absolute links and PNG images (the 0.1.0 page had broken relative links and a
  broken chart).
- Install command now includes the prebuilt llama.cpp index, so no C++ compiler is needed on
  Windows or Linux: `pip install "quantgap[llamacpp]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`.
  macOS GGUF scoring is not verified (the prebuilt macOS wheel is currently broken).
- Free GPU notebook is for Google Colab (it was Kaggle).
- Launch video, real sample results and chart, Hugging Face viewer fix for packs.
- The 0.1.0 source package accidentally shipped an internal planning file. 0.1.1 does not.

## 0.1.0

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
