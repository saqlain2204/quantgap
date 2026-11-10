# Make and publish a pack

A **pack** is the reference model's answers on the probe set. You make it once. Everyone who
scores a quant of that model reuses it.

## Quick version

```bash
pip install "quantgap[pack]"
quantgap pack Qwen/Qwen2.5-0.5B-Instruct --revision <full commit sha> --license apache-2.0
quantgap packs show packs/Qwen2.5-0.5B-Instruct
```

Always pass `--revision` (the model's commit sha from its Hub page). It pins the exact weights, so
anyone can rebuild the pack.

Upload it:

```bash
export HF_TOKEN=hf_...            # a token with write access
quantgap pack Qwen/Qwen2.5-0.5B-Instruct --revision <sha> --push your-org/Qwen2.5-0.5B-Instruct-pack
```

The dataset card is written for you. It carries the tag `quantgap-pack`, so
`quantgap packs list` finds it.

## What size model can I pack?

The MVP targets models up to about 32B parameters. Rough guide for memory:

| Model size | Memory for the reference run |
|---|---|
| up to 3B | CPU is fine, but slow (the 0.5B example below took minutes, not hours) |
| 7B to 14B | a 24 GB GPU, or a Kaggle T4 x2 (32 GB) with `device_map="auto"` |
| 27B to 32B | Kaggle T4 x2 with CPU offload: works, but slow |

These are estimates, not measurements. Real timings will be added once more packs exist.
The `bfloat16` weights have to fit in GPU memory (plus a few GB), or `accelerate` offloads layers
to CPU.

## Kaggle (free GPU)

`notebooks/make_pack_kaggle.ipynb` installs quantgap, runs `quantgap pack`, and uploads the
result. Turn on GPU T4 x2 and Internet in the notebook settings, and add `HF_TOKEN` as a Kaggle
secret.

## Licensing: read this before you upload

A pack holds **model outputs** (log probabilities) on openly licensed texts. It does not hold
weights. Still:

- Put the reference model's license in the pack (`--license`). It goes on the dataset card.
- **Do not publish a pack for a model whose license forbids sharing its outputs.**
- The probe texts are CC BY-SA, MIT, Apache-2.0 or CC0. See `SOURCES.md`.

## What is inside

```
manifest.json        model, revision, tokenizer fingerprint, probe set hash, file hashes
probes.parquet       the token ids of each probe text and the score mask
topk-00000.parquet   top-k ids and log probabilities for each scored position
greedy.parquet       the reference's own short generations (for the damage checks)
```

The pack id is the first 16 hex characters of the SHA-256 of the canonical manifest. The manifest
holds a hash of every file, so changing one byte anywhere changes the id, and `quantgap` refuses
to use a file that does not match.
