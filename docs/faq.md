# FAQ

## Why are my numbers a bit different from llama.cpp's KLD?

llama.cpp's `--kl-divergence` sums over the whole vocabulary. quantgap keeps the reference's top
32 guesses and lumps the rest into one bucket, so its value is a **lower bound** of the true KLD
(see [method.md](method.md)). Rankings agree. Absolute numbers are slightly lower. Different probe
texts change the numbers too. Compare quantgap numbers with other quantgap numbers.

## Why not Ollama?

To measure drift we feed fixed text and read the model's probabilities for **every position of
that text**. Ollama's API does not return those (it gives logprobs for generated tokens only).
You can still score the file Ollama downloaded: run `ollama show --modelfile NAME`, take the path
on the `FROM` line (a GGUF blob), and pass it to `quantgap score`.

## Is my data sent anywhere?

No. quantgap has no telemetry. It talks to the Hugging Face Hub when you ask it to download a
pack or a quant, or to upload a pack or result. With `--base-url` it talks to the server you name.
Scoring a local file on a local pack makes no network calls.

## Do I need the original model?

No. That is the point. Someone runs the original once and publishes a pack. You only need the pack
and your quant.

## My quant says "looks like a fine-tune".

The GGUF says it was made from a different model than the pack's. Then the drift includes the
fine-tuning, not only quantization. Score it against a pack of its own base, or make a pack for
the fine-tune.

## Why does it refuse with "tokenizer ... failed"?

The quant must use the same token ids as the pack. quantgap encodes a fixed test string and
compares ids, and checks BOS and EOS. Different tokenizer means the numbers mean nothing, so it
stops. It never "fixes" a mismatch.

## What is `approximate`?

An OpenAI-compatible server returns only its own top-N tokens (often 20). Reference tokens missing
from that list are counted in the tail bucket. The number is still a valid lower bound, but a
coarser one. Leaderboard rows carry an `approx` marker.

## Why is `--ctx` required to be at least 600?

The longest probe text is 512 tokens, plus a few more for special tokens. quantgap refuses to run
with less than the longest text plus 8, and tells you the number to use.

## Can I score a model that is bigger than my memory?

llama.cpp can offload layers to the CPU (`--n-gpu-layers`), so it works, slowly. For very large
mixture-of-experts models, make sure the packs exist first. Packs above about 32B parameters need
a big machine (see [packs.md](packs.md)).

## Can I use this as a regression test for my inference engine?

Yes. Run `quantgap score --backend openai --base-url ... --json out.json` in CI and fail if
`kld_mean` rises above your baseline. A reusable GitHub Action is on the wish list.
