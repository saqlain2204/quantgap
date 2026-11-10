# Decisions

Choices made while building quantgap, and facts checked at the start. One short
paragraph each.

## M0: API facts

**llama-cpp-python exposes per-position logits.** Checked by running it (v0.3.36, Windows CPU
wheel). `Llama(model_path, logits_all=True)`, then `llm.eval(tokens)`, fills `llm.scores`, a numpy
array of shape `(n_ctx, n_vocab)`. Row `i` holds the logits after reading token `i`, so it predicts
token `i + 1`. The attribute is `scores` (not `_scores`). Tests compare a tiny F16 GGUF with the
Hugging Face model and get KLD below 1e-3.

**vLLM `prompt_logprobs`: taken from the docs, not run against a live GPU.** `/v1/completions`
accepts `prompt` as a list of token ids and `prompt_logprobs: N`. The reply has
`choices[0].prompt_logprobs`: one entry per prompt token, entry 0 is null, other entries are dicts
`{"<token id>": {"logprob", "rank", "decoded_token"}}` holding the actual token plus the server's
top-N. N is capped by the server's `--max-logprobs` (default 20). The adapter asks for the
configured N and steps down (20, 10, 5, 1) if the server answers 400. Verified only against a fake
server (`tests/test_openai_compat.py`) that follows this shape. Please report differences you find
on a real vLLM.

**MLX: written from the mlx-lm API, not run on a Mac.** `mlx_lm.load` returns `(model, tokenizer)`
and `model(mx.array([ids]))` returns logits `(1, T, V)`. The author had no Apple silicon machine,
so `quantgap/backends/mlx.py` is untested. It shares all maths with the llamacpp and hf backends
(`gather_from_logits`), so the risk is in loading and generation only.

## Format and maths

**Pack log probabilities are stored as float32, not float16.** The first design used float16. With float16
the stored reference numbers carry a rounding error of about 1e-4 that adds a first-order noise
term to KLD, so scoring the reference against its own pack would not reach the 1e-6 self-test
target. float32 costs about 20 MB for 80,000 positions at k=32, still inside the 10 to 30 MB range.
`quantgap pack --logp-dtype float16` is available and the manifest records the choice.

**Backends gather the quant's probability for the reference's top-k ids.** The shared helper does
a float64 log-softmax in chunks of 64 positions so 150k-token vocabularies stay cheap.

**Ids a server did not report are moved into the tail bucket for both distributions.** The first design counted them in the tail. We do this on both sides (reference and quant) so the result stays
a valid coarsening, and therefore a valid lower bound, and never negative.

**Tail clamp counting.** `q_tail` is floored at 1e-12 silently. A clamp is counted only when the
quant's tail comes out below -1e-6 (probabilities adding up to more than 1) or when a head
probability is below 1e-12. Counting every floored tail would flag healthy models.

**The wrong-script check has no reference count.** The reference defines which scripts are
expected, so it cannot disagree with itself. `wrong_script_ref` is always 0 and is kept in the file
so the layout matches the other damage numbers.

**Loop and wrong-script rates are counts out of the prompts run.** Quick mode runs 2 prompts per
group instead of 8.

## Probe set

**Probe set v1 has 160 rows, but not exactly 6 equal groups.** general 30, chat 30, code 28, math
28, multilingual 32 (8 each of Hindi, Chinese, Spanish, Arabic), json 12. 8 per group are used for
greedy damage checks (48 total).

**Wikipedia text comes from the `wikimedia/wikipedia` dataset on the Hugging Face Hub (20231101
snapshot), not the live Wikipedia API.** The live API refused the builder with HTTP 403. Same CC
BY-SA license. Articles are picked at fixed offsets, so the choice is arbitrary but repeatable.

**Chat prompts come from Dolly-15k (CC BY-SA 3.0)**, rows without context text and with a reply of
250 to 1500 characters. Math is GSM8K train (MIT) with the `<<calc>>` annotations removed.

**Code rows are this project's own Apache-2.0 source plus 6 hand-written snippets in other
languages.** The checked-in `probes_v1.jsonl` is the source of truth. Re-running `build.py` later
would pick up changed source files and different Wikipedia text, so do not expect identical output.

**Chat score mask uses character offsets.** A BPE token can straddle the end of the prompt and the
start of the reply (" The" glues the space from the template to the first word). That token is
treated as context and is not scored.

**Tokenizer hash.** `tokenizer.sha256` is the hash of `tokenizer.backend_tokenizer.to_str()`, not
the raw `tokenizer.json` file bytes, so packs made with different library versions can differ.
It is informational. The id check uses `check_ids`.

**The tokenizer check string contains no special-token text.** Special tokens vary by model and
are compared by id (BOS, EOS) instead.

## Behaviour

**Reference runs on CPU use float32, on GPU bfloat16.** The dtype used is recorded in the manifest.
bfloat16 on CPU is very slow and float32 is the more accurate reference.

**Hash of quant files.** For `hf://` GGUF files the SHA-256 comes from the Hub's LFS metadata
instead of hashing the download again. Local files are hashed on load.

**Split GGUF files** (`-00001-of-0000N.gguf`) are not supported yet. A glob keeps only part 1 so
nothing is scored twice, but llama.cpp loading of split files has not been tested here.

**`--quick` also runs fewer damage prompts** (2 per group) and marks the result not submittable.

**Auto pack lookup.** The pack for a quant is found through the GGUF key
`general.base_model.0.repo_url`, by listing datasets tagged `quantgap-pack` under the packs
organisation and comparing `manifest.model.repo`. Many GGUFs do not write this key. Then the user
must pass `--pack`.

**No telemetry.** Nothing is ever sent anywhere except to the Hugging Face Hub (downloads, and
uploads you ask for) and the server you point `--base-url` at.

## Not done yet (honest list)

- The Hugging Face organisations `quantgap-packs` and `quantgap` are not created by this repo. Until
  they exist, `packs list`, automatic pack lookup and `submit` need `--pack` / `--results-repo`
  pointing at something you own.
- No seeded packs or leaderboard at the spec's launch size (4 packs, 40+ results). One real pack
  and a handful of results are in `docs/RESULTS.md`, made on one CPU machine.
- MLX adapter and a live vLLM server are untested (see above).
- Split GGUF files, Docker image publishing and PyPI publishing have not been exercised.
