<p align="center">
  <img src="https://raw.githubusercontent.com/saqlain2204/quantgap/main/assets/logo-banner.png" alt="quantgap: how much did this quant lose?" width="560">
</p>

<h1 align="center">quantgap: how much did this quant lose?</h1>

<p align="center">
  <a href="https://github.com/saqlain2204/quantgap/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/saqlain2204/quantgap/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://huggingface.co/quantgap-packs"><img alt="Packs on Hugging Face" src="https://img.shields.io/badge/packs-Hugging%20Face-f59e0b"></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3b82f6">
  <a href="https://github.com/saqlain2204/quantgap/blob/main/LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-3b82f6"></a>
</p>

**quantgap measures how far a quantized model drifts from the original, on your own machine, without running the original.**

<p align="center">
  <a href="https://github.com/saqlain2204/quantgap/blob/main/docs/launch.mp4" title="Watch the 87-second launch video">
    <img src="https://raw.githubusercontent.com/saqlain2204/quantgap/main/docs/launch.gif" alt="quantgap launch video: what it does and how to use it" width="720">
  </a>
  <br>
  <sub>Click for the full <a href="https://github.com/saqlain2204/quantgap/blob/main/docs/launch.mp4">87-second launch video</a> (1080p): what it does, how to use it, real results.</sub>
</p>

```text
Which quant should I download?   Q4_K_M?  UD-Q4_K_XL?  IQ3_XXS?  NVFP4?
```

People answer that with feelings. quantgap answers it with a number: how much each quant moved away
from the real model, measured the same way for every publisher and every format.

## Quickstart

```bash
pip install "quantgap[llamacpp]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# how much did this one file lose?
quantgap score hf://<publisher>/<model>-GGUF/<file>.gguf

# score many quants of one model, draw a chart, and get a pick for your memory
quantgap compare --repo <publisher>/<model>-GGUF --glob "*Q4*" --budget 16GB --chart chart.svg
```

The extra index gives pip a ready-made llama.cpp for Windows and Linux, so you do not need a C++
compiler. **macOS:** that ready-made file is currently broken, and we have not tested GGUF scoring on a
Mac. If you try it, build llama-cpp-python from source first
(`CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python`, needs the Xcode command line tools),
then `pip install quantgap`, and tell us how it went.

**No GPU?** Score quants on a free Colab T4 instead:
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saqlain2204/quantgap/blob/main/notebooks/score_quants_colab.ipynb) it took about 3 minutes
per quant of a 0.5B model. (The notebook was tested on a free T4 runtime on 2026-11-10.)

You need a **pack** for the model (the original's answers on a fixed set of texts, made once and
shared). Pass one with `--pack`, or let quantgap look one up on the Hub. See
[Make a pack for your own model](#make-a-pack-for-your-own-model).

## Sample results (real, small model)

Model: `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0). Its pack is public at
[quantgap-packs/Qwen2.5-0.5B-Instruct](https://huggingface.co/datasets/quantgap-packs/Qwen2.5-0.5B-Instruct),
so `quantgap score hf://bartowski/Qwen2.5-0.5B-Instruct-GGUF/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` finds it by itself.
The numbers below were measured by quantgap against a pack made from the original weights. Hashes and settings:
[docs/RESULTS.md](https://github.com/saqlain2204/quantgap/blob/main/docs/RESULTS.md).

| Publisher | Quant | Size | KLD mean (95% interval) | Worst 1% | Top-1 | PPL ratio | Loops (quant / original) | Wrong script | JSON ok |
|---|---|---|---|---|---|---|---|---|---|
| bartowski | Q8_0 | 531.1 MB | 0.0025 (0.0023 to 0.0026) | 0.021 | 97.1% | 1.003 | 27 / 23 | 0 | 7 / 8 |
| bartowski | Q6_K | 505.7 MB | 0.0043 (0.0039 to 0.0047) | 0.039 | 96.6% | 1.003 | 22 / 23 | 0 | 8 / 8 |
| bartowski | Q5_K_M | 420.1 MB | 0.0229 (0.0205 to 0.0258) | 0.249 | 92.3% | 1.023 | 24 / 23 | 0 | 8 / 8 |
| Qwen | Q8_0 | 675.7 MB | 0.0275 (0.0251 to 0.0304) | 0.284 | 91.4% | 1.035 | 27 / 23 | 0 | 8 / 8 |
| bartowski | Q4_K_M | 397.8 MB | 0.0337 (0.0303 to 0.0382) | 0.325 | 90.6% | 1.035 | 29 / 23 | 0 | 8 / 8 |
| bartowski | Q4_K_S | 385.5 MB | 0.0429 (0.0384 to 0.0485) | 0.453 | 89.3% | 1.042 | 22 / 23 | 0 | 8 / 8 |
| Qwen | Q5_K_M | 522.2 MB | 0.0498 (0.0451 to 0.0551) | 0.546 | 88.8% | 1.057 | 24 / 23 | 0 | 8 / 8 |
| Qwen | Q4_K_M | 491.4 MB | 0.0647 (0.0587 to 0.0716) | 0.691 | 87.2% | 1.066 | 31 / 23 | 0 | 8 / 8 |
| bartowski | Q3_K_M | 355.5 MB | 0.0724 (0.0648 to 0.0815) | 0.848 | 86.5% | 1.072 | 27 / 23 | 0 | 8 / 8 |
| bartowski | IQ4_XS | 349.4 MB | 0.0837 (0.0748 to 0.0943) | 1.029 | 86.0% | 1.076 | 27 / 23 | 0 | 7 / 8 |
| Qwen | Q3_K_M | 432.0 MB | 0.1109 (0.1002 to 0.1235) | 1.323 | 83.9% | 1.115 | 26 / 23 | 0 | 8 / 8 |
| bartowski | IQ3_XS | 338.6 MB | 0.1258 (0.1140 to 0.1399) | 1.398 | 82.6% | 1.132 | 31 / 23 | 0 | 8 / 8 |
| bartowski | Q2_K | 338.6 MB | 0.1706 (0.1561 to 0.1872) | 1.831 | 79.6% | 1.197 | 34 / 23 | 0 | 8 / 8 |
| Qwen | Q2_K | 415.2 MB | 0.1761 (0.1605 to 0.1954) | 1.958 | 79.5% | 1.214 | 29 / 23 | 0 | 8 / 8 |
| bartowski | IQ2_M | 328.6 MB | 0.2604 (0.2404 to 0.2829) | 2.453 | 74.9% | 1.310 | 34 / 23 | 0 | 8 / 8 |

Full mode: all 160 texts and 48 generations per quant, llama.cpp on a free Colab Tesla T4 GPU. Lower KLD is better. The original loops in 23 of 48 prompts under plain greedy decoding (common for a 0.5B model), so read the loops column relative to that.

![quality versus size](https://raw.githubusercontent.com/saqlain2204/quantgap/main/docs/chart.png)

**What the data shows:** bartowski's files form a steady frontier. For the same label, the official Qwen file drifted further in 5 of 5 matched pairs (for example Q4_K_M: 0.0647 for Qwen's 491.4 MB file against 0.0337 for bartowski's 397.8 MB file, about 1.9 times as far and 94 MB bigger). We did not investigate why. A label like `Q4_K_M` names a recipe, not one fixed result, which is the point of measuring.

Same-machine check: the same Q8_0 file on the same subset of texts gave a KLD of 0.002720 on a laptop CPU
and 0.002694 on the Colab GPU, a difference of 0.95%. This is a small model and a demonstration of the tool, not a verdict on
these publishers.

## How it works

```
        (once per model: big machine, Colab, or HF Jobs)
 original model ──► quantgap pack ──► pack (small files) ──► Hugging Face Hub
                                                                  │
        (you, on your own laptop)                                 ▼
 quant file / server ──► backend ──► per-position scores ──► quantgap score ◄── pack
                                                                  │
                                                  table + chart + result JSON
                                                                  │
                                            quantgap submit ──► pull request ──► leaderboard
```

1. **Pack (once).** `quantgap pack` runs the full-size model over 160 fixed texts and saves its
   top 32 next-token guesses at every position. That is about 20 MB, not many GB. **Teacher
   forcing** means we feed the same text to both models and compare their guesses at every
   position. Nobody generates anything, so it is fair and fast.
2. **Score (anyone).** `quantgap score` runs your quant on the same texts on your own machine
   (llama.cpp, MLX, Transformers, or any vLLM server) and compares.
3. **Share.** `quantgap submit` sends the result, with the SHA-256 of the quant file and of the
   pack, so anyone can re-run it and get the same number.

## What the numbers mean

**Lower KLD is better.**

| Number | In plain words |
|---|---|
| **KLD mean** | One number for how far the quant's next-token probabilities are from the original's. 0 means identical. A 95% interval comes with it. |
| **KLD worst 1%** | The 99th percentile. A mean can hide rare big failures. This catches them. |
| **Top-1 agreement** | How often the quant's best guess is the same as the original's. |
| **Perplexity ratio** | How much more surprised the quant is by real text. 1.0 means no change. |
| **Loops** | Short greedy generations where the model repeats itself. Shown next to the original's count. |
| **Wrong script** | A Chinese, Cyrillic, Arabic or Hindi character appears in English, code, math or JSON text where the original had none. |
| **Valid JSON** | Of 8 prompts that ask for a JSON object, how many parse. Shown next to the original's count. |

Numbers are also given for each group of texts: general, chat, code, math, multilingual
(Hindi, Chinese, Spanish, Arabic) and JSON. The maths is explained in plain words in
[docs/method.md](https://github.com/saqlain2204/quantgap/blob/main/docs/method.md).

## What quantgap does not tell you

- **It is not a skill benchmark.** It measures drift from the original, not how good the model is
  at your task. Less drift is not always better for you.
- **It is not a speed benchmark.** Use `llama-bench` or [llmfit](https://github.com/AlexsJones/llmfit) for speed.
- **The KLD is a lower bound.** To keep packs small we keep the original's top 32 guesses and
  lump everything else into one bucket. That can only make two models look closer, never further
  apart. It ranks quants well. Its absolute values are a little lower than llama.cpp's
  full-vocabulary numbers. Compare quantgap numbers with quantgap numbers.
- **Ollama is not supported yet.** Ollama does not return the model's scores for the input text.
  You can score the GGUF file Ollama already downloaded (`ollama show --modelfile NAME` prints
  its path).
- **Big models need someone with a big machine.** The MVP targets models up to about 32B
  parameters.

## Make a pack for your own model

```bash
pip install "quantgap[pack]"
quantgap pack <org>/<model> --revision <full commit sha> --license <license> --push <you>/<name>
```

Free option: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/saqlain2204/quantgap/blob/main/notebooks/make_pack_colab.ipynb)
runs it on a free Colab T4 GPU. Details, memory guidance and licensing rules are in [docs/packs.md](https://github.com/saqlain2204/quantgap/blob/main/docs/packs.md).
Check that the model's license allows sharing its outputs first.

You fine-tuned a small model and want to know how much Q4_K_M hurts it? Make a pack for your
fine-tune, then score the GGUF against it.

## Backends

| Backend | For | Install | Status |
|---|---|---|---|
| `llamacpp` | GGUF on CPU and CUDA | `pip install "quantgap[llamacpp]" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu` | tested on Windows and Linux with real GGUF files; **macOS and Metal not verified** |
| `openai` | any OpenAI-compatible server that returns prompt logprobs (vLLM: NVFP4, AWQ, GPTQ, FP8) | built in, use `--base-url` | tested against a fake server that follows the vLLM docs, not a live GPU |
| `hf` | Transformers, bitsandbytes | `pip install "quantgap[hf]"` | tested here (the reference path) |
| `mlx` | Apple silicon | `pip install "quantgap[mlx]"` | **written but not run on a Mac yet** |

```bash
# a vLLM server
quantgap score openai://Qwen2.5-0.5B-Instruct-NVFP4 --base-url http://localhost:8000/v1 --pack <pack>
```

A server only reports its own top-N tokens, so those results are marked **approximate**.

## Submit a result

The public leaderboard site is built and waiting. It opens when there are full results (a `--quick`
result cannot be submitted), so yours can be one of the first.

```bash
quantgap score ./my-quant.gguf --pack <pack>          # writes quantgap-results/<name>.json
quantgap submit quantgap-results/<name>.json --dry-run # shows exactly what would be sent
export HF_TOKEN=hf_...
quantgap submit quantgap-results/<name>.json           # opens a pull request
```

`--quick` results are for a fast first look and cannot be submitted.

## Use it in CI

```bash
quantgap score --backend openai --base-url "$URL" --model-name "$NAME" --json out.json openai://"$NAME"
python -c "import json,sys; sys.exit(json.load(open('out.json'))['metrics']['kld_mean'] > 0.02)"
```

Fail the build if your inference engine fork starts to drift.

## Commands

```
quantgap pack MODEL_REPO [--revision SHA] [--k 32] [--device auto|cuda|cpu] [--out DIR] [--push ORG/NAME]
quantgap packs list | show PACK_REF
quantgap score QUANT [--pack PACK_REF] [--backend auto|llamacpp|mlx|openai|hf] [--quick] [--json OUT]
quantgap compare QUANT... | --repo ORG/REPO --glob "*Q4*.gguf" [--budget 16GB] [--chart out.svg]
quantgap submit RESULT.json [--dry-run]
quantgap inspect RESULT.json
quantgap doctor [--selftest]
```

`quantgap doctor` tells you which backends are installed and which extra to add.

## Comparison with other tools

| Project | What it does | What quantgap adds |
|---|---|---|
| llama.cpp `llama-perplexity --kl-divergence` | Built-in KLD of a GGUF against a saved logits file from the BF16 model. | You must run the BF16 model yourself and the logits file is huge. quantgap shares small packs, works beyond GGUF, and adds a report. We learned from it: it is the prior art. |
| [kld-sweep](https://github.com/cmhamiche/kld-sweep) | Loops `llama-perplexity` over a folder of GGUFs, with CSV and a chart. | Same idea, nice and simple. quantgap removes the need for a local BF16 run and adds damage checks and a shared leaderboard. |
| [mlx-kld](https://github.com/asher/mlx-kld) | KLD for MLX and GGUF with a local teacher cache. | quantgap makes the "teacher cache" a shareable pack and covers more backends. |
| [GGUF-Tool-Suite](https://github.com/Thireus/GGUF-Tool-Suite) | Builds custom mixed quants to a target size. | It makes quants, quantgap judges them. They fit together. |
| [quant-fidelity-frontier](https://github.com/delirious6423/quant-fidelity-frontier) | Collects numbers that publishers reported. | quantgap measures with one method and records hashes. |
| [llmfit](https://github.com/AlexsJones/llmfit) | Which models fit your hardware, and how fast. | quantgap is the measured-quality half. |

## FAQ

**Why are my numbers a bit different from llama.cpp's KLD?** quantgap keeps the top 32 guesses and
a tail bucket, so it is a lower bound. See [docs/faq.md](https://github.com/saqlain2204/quantgap/blob/main/docs/faq.md).

**Why not Ollama?** Ollama does not return the model's scores for the input text, so we cannot
compare fairly. Score the GGUF file it downloaded instead.

**Is my data sent anywhere?** No. quantgap has no telemetry. It only talks to the Hugging Face Hub
(downloads, and uploads you ask for) and to a server you name with `--base-url`.

**Why does it stop with a tokenizer error?** The quant must use the same token ids as the pack.
quantgap checks and never fixes a mismatch silently.

More in [docs/faq.md](https://github.com/saqlain2204/quantgap/blob/main/docs/faq.md).

## Not yet

- Scoring through Ollama (it does not return the model's scores for the input text).
- Task benchmarks such as MMLU or coding pass rates. quantgap measures drift, not skill.
- Packing models above about 32B parameters (memory limits on free hardware).
- Vision and audio models.
- Speed benchmarks (use `llama-bench` or llmfit).
- KV-cache quantization effects.

## Development

```bash
git clone https://github.com/saqlain2204/quantgap && cd quantgap
pip install -e ".[dev]" llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
python scripts/make_tiny_model.py      # tiny model + tiny GGUF for tests (also built by the tests)
python scripts/make_video.py            # re-render docs/launch.mp4 from docs/results (needs ffmpeg)
python -m pytest
python -m ruff check . && python -m ruff format --check .
```

See [CONTRIBUTING.md](https://github.com/saqlain2204/quantgap/blob/main/CONTRIBUTING.md) for how to add a backend and how to submit a pack.

## Credits

- **llama.cpp** and its `--kl-divergence` tooling are the prior art this builds on.
- **[kld-sweep](https://github.com/cmhamiche/kld-sweep)** and **[mlx-kld](https://github.com/asher/mlx-kld)** are related work.
- Probe texts come from Wikipedia (CC BY-SA 4.0), Dolly-15k (CC BY-SA 3.0) and GSM8K (MIT). See
  [`SOURCES.md`](https://github.com/saqlain2204/quantgap/blob/main/src/quantgap/probes/data/SOURCES.md).

## License

Code: Apache-2.0. Results dataset: CC BY 4.0. Each pack's dataset card states the license of the
original model and that the pack holds model outputs (log probabilities), not weights.
