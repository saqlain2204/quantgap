# Sample results: Qwen2.5-0.5B-Instruct

Real measurements made by quantgap 0.1.0. Nothing here is estimated or copied from a publisher.

- **Reference model:** `Qwen/Qwen2.5-0.5B-Instruct` at commit `7ae557604adf67be50417f59c2c2f167def9a775` (Apache-2.0), run in float32 on CPU.
- **Pack (public):** https://huggingface.co/datasets/quantgap-packs/Qwen2.5-0.5B-Instruct, `Qwen2.5-0.5B-Instruct@f499c4436fa8e370`, probe set v1, k=32, 49,013 scored positions, built 2026-11-10T09:37:57Z.
- **Scoring:** llama-cpp-python 0.3.36, `--n-gpu-layers 0`, `--quick`, on Intel64 Family 6 Model 142 Stepping 10, GenuineIntel (Windows).
- **Not submittable:** these are `--quick` results (first 25% of each group), so they are a first look, not leaderboard entries.

| Publisher | Quant | Size | KLD mean (95% interval) | Worst 1% | Top-1 | PPL ratio | Loops (quant / original) | Wrong script | JSON ok |
|---|---|---|---|---|---|---|---|---|---|
| bartowski | Q8_0 | 531.1 MB | 0.0027 (0.0024 to 0.0031) | 0.027 | 97.1% | 1.003 | 5 / 6 | 0 | 2 / 2 |
| bartowski | Q6_K | 505.7 MB | 0.0069 (0.0057 to 0.0083) | 0.064 | 95.5% | 1.008 | 8 / 6 | 0 | 2 / 2 |
| bartowski | Q5_K_M | 420.1 MB | 0.0241 (0.0190 to 0.0298) | 0.264 | 92.3% | 1.024 | 5 / 6 | 0 | 2 / 2 |
| bartowski | Q4_K_M | 397.8 MB | 0.0389 (0.0316 to 0.0478) | 0.376 | 89.9% | 1.041 | 6 / 6 | 0 | 2 / 2 |
| Qwen | Q4_K_M | 491.4 MB | 0.0787 (0.0627 to 0.0950) | 0.807 | 85.8% | 1.064 | 7 / 6 | 0 | 2 / 2 |
| bartowski | Q3_K_M | 355.5 MB | 0.0828 (0.0672 to 0.1009) | 0.852 | 85.9% | 1.073 | 7 / 6 | 0 | 2 / 2 |
| bartowski | Q2_K | 338.6 MB | 0.2052 (0.1672 to 0.2386) | 2.080 | 77.8% | 1.200 | 9 / 6 | 0 | 2 / 2 |

Quick mode (first 25% of each group, 12 greedy prompts), llama.cpp on CPU, one machine. Lower KLD is better. Loops are counted out of 12 prompts, so differences of one or two are noise.

![quality versus size](chart.svg)

**One finding worth a second look:** the two files labelled `Q4_K_M` are not the same quality. The official Qwen file (491 MB) drifted about twice as far (KLD 0.0787) as bartowski's (398 MB, KLD 0.0389), and the intervals do not overlap. We did not investigate why. A label like `Q4_K_M` names a recipe, not one fixed result, which is the point of measuring.

## How to read it

KLD says how far each quant's next-token probabilities moved from the original's. It climbs steadily as the file gets smaller: Q8_0 and Q6_K are close to the original, Q4_K_M is a visible step away, and Q2_K has drifted a lot. See [method.md](method.md) for what the numbers mean and why KLD here is a lower bound.

The loop counts barely move, and the original itself loops in 6 of 12 prompts under plain greedy decoding, which is common for a 0.5B model. With only 12 prompts this check is weak in quick mode. Wrong-script and JSON checks found nothing to flag on this model.

## Quant files (SHA-256)

The seven result files are in [`results/`](results/).

- `25130a98aa782284a7dabea0c23245b2fd371ed47244e79d78b8ec23245fdf96` bartowski Q8_0
- `32c14c29a44712c02e29d5c2605593ece92ccb7a4358f56016a42b151434c842` bartowski Q6_K
- `a0a413dcbb4676f21d4c951b98a393324694edb1a20a4f9547d1de8d2919ff3b` bartowski Q5_K_M
- `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653` bartowski Q4_K_M
- `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` Qwen Q4_K_M
- `e14591771b0a3070c50ff11ee6055ffbb25f0bf50b6f9474ec4a16ff4ace8df6` bartowski Q3_K_M
- `c929ba8be98f584d0872923e373bda0da8a261eae05dd362590c9792adfc76d2` bartowski Q2_K

## Reproduce

```bash
pip install "quantgap[pack,llamacpp]"
quantgap pack Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --license apache-2.0 --device cpu
python scripts/seed_launch.py scripts/seed_list_sample.json --quick --n-gpu-layers 0
```

A new pack will have a different id (it holds a timestamp), so numbers should match within a small tolerance, not byte for byte. These results were not yet re-run on a second machine.
