# Contributing

Thanks for helping. quantgap is small on purpose. Short sentences below.

## Set up

```bash
git clone https://github.com/saqlain2204/quantgap && cd quantgap
uv sync --all-extras            # or: pip install -e ".[dev]" llama-cpp-python
python -m pytest                # fast tests (builds a tiny model; about a minute)
python -m pytest -m slow        # slow tests (real downloads); nightly in CI
python -m ruff check . && python -m ruff format --check .
```

The tests build a tiny random Llama model and tiny GGUF files (`scripts/make_tiny_model.py`), so no
download is needed. If `llama-cpp-python` is missing, the GGUF tests are skipped.

## Add a backend

A backend is one file in `src/quantgap/backends/` with a class that follows the `Backend`
protocol in `base.py`:

| Method | What it does |
|---|---|
| `load(source, settings)` | open the quant |
| `tokenizer_fingerprint()` | vocab size, BOS/EOS ids, ids of the fixed check string |
| `prompt_logprobs(token_ids, positions, gather_ids)` | teacher forcing: the model's log probability of each id in `gather_ids[row]`, of the actual next token, and its own top-1 |
| `generate_greedy(token_ids, max_new)` | for the damage checks |
| `decode(ids)` | ids to text |
| `version()`, `close()` | metadata, cleanup |

If your engine gives full logits, call `gather_from_logits` (shared float64 maths). If it gives only
a top-N, return `NaN` for ids it did not report and set `approximate=True`. Then register the name
in `backends/__init__.py` and add a test. Do not change the scoring maths for one backend. Adapt
the adapter.

Please also add a line to `docs/DECISIONS.md` saying how you checked the engine's API.

## Submit a pack

See [docs/packs.md](docs/packs.md). Check the reference model's license first. Do not publish a
pack for a model whose license forbids sharing its outputs.

## Submit a result

`quantgap submit result.json` opens a pull request on the results dataset. The result must not be
`--quick`, and must carry the SHA-256 of the quant file. Someone else should be able to re-run it
and get a `kld_mean` within about 2%.

## Style

- Plain words in docs, CLI messages and comments. Explain a technical word once.
- Write any design choice that is not obvious into `docs/DECISIONS.md`, one short paragraph.
- Never invent numbers. README tables come from real runs only.
- Keep the default install free of torch. Heavy imports stay inside functions.
