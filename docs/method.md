# How quantgap measures drift

Short sentences, plain words. Each technical word is explained once.

## The setup

A **quant** is a smaller copy of a model that stores its numbers with fewer bits. It uses less
memory but can get worse. The **reference model** is the original full-size copy.

We never ask "is the quant good at tasks?" We ask "how far did the quant move away from the
reference?" That is a cleaner question. It has no right answers to argue about, and it works the
same for every model.

## Teacher forcing

We take a fixed set of texts (the **probe set**) and feed the same text to both models. At every
position we compare what each model thinks the **next token** should be. Nobody generates
anything. Because both models read the same text, the comparison is fair, and it is fast: one pass
instead of a long generation.

The reference model reads the probe set **once**. We save its answers in a **pack**. Everyone
else scores their quant against the pack and never needs the big model.

## What is saved in a pack

At every scored position the model gives a score to every token in its vocabulary. That is too
much to store (150,000 numbers per position), so we keep only the **top 32 guesses** and their
log probabilities. We also keep the log probability of the token that really came next.
80,000 positions at k=32 is about 20 MB. A log probability is the natural log of a probability.

## KLD, in plain words

**KL divergence (KLD)** is one number that says how far apart two sets of probabilities are. 0
means identical. Bigger means more drift. At each position:

1. The reference gives probabilities `p` to its 32 best tokens. Whatever is left over,
   `p_tail = 1 - sum(p)`, goes into one extra bucket called the **tail**.
2. The quant gives probabilities `q` to the same 32 tokens. Its leftover is `q_tail`.
3. We compute

```
KL_k = sum over the 32 tokens of  p * log(p / q)   +   p_tail * log(p_tail / q_tail)
```

then average over all scored positions.

### Why this is a lower bound

Merging many tokens into one bucket can only make two distributions look **closer**, never
further apart. So `KL_k` is always less than or equal to the true KLD over the whole vocabulary.
With 32 tokens the tail is usually tiny, so the gap is small. Tests check this numerically
(`tests/test_kld.py`, T3): on 1,000 random pairs, `KL_k <= exact KLD` always, and the gap shrinks
as k grows.

**What it means for you.** Good for ranking quants of one model against each other. Do not
compare its absolute values with llama.cpp's `--kl-divergence` numbers: llama.cpp sums over the
full vocabulary, so its values are a little higher.

### Servers that report only their own top-N

An OpenAI-compatible server (vLLM) tells us its own top-N tokens, not its probability for each of
the reference's 32. If a reference token is missing from the server's list, we move that token
into the tail bucket for **both** models. The result is still a valid lower bound. It is a coarser
one, so the result is marked **approximate**.

## The other numbers

- **Top-1 agreement.** How often the quant's single best guess is the same as the reference's.
- **Perplexity ratio.** `exp(mean(-log q_target)) / exp(mean(-log p_target))` where `*_target` is
  the probability given to the token that really came next. 1.0 means no change. Above 1.0 means
  the quant is more surprised by real text.
- **Worst 1% / worst 0.1% of KLD** (99th and 99.9th percentile of per-position KLD). A mean can
  hide rare big failures. These catch them.
- **Confidence interval.** A bootstrap over **sequences**, 1,000 resamples, seed 0. We resample
  whole sequences, not single positions, because positions inside one text are not independent.
  Resampling positions would make the interval too narrow.

## Numerical rules

- All maths in float64.
- The quant's probabilities are floored at `1e-12` before taking a log. The number of floors is
  reported. A high count means something is broken (wrong tokenizer, damaged file).
- If `p_tail < 1e-9` the tail term is skipped (avoids `0 * log 0`).
- The quant's log-softmax subtracts the maximum first (stable).

## Visible damage checks

KLD can look small while the model still does something obviously wrong. After the KLD pass we run
48 short greedy generations (8 per group, at most 256 tokens) and count:

- **Loops.** A text loops if an 8-token run repeats 4 or more times, or the last 64 tokens use
  fewer than 12 different tokens. We show the quant's count **next to the reference's**, because
  the reference loops too sometimes.
- **Wrong script.** In English, code, math and JSON prompts, a text is flagged if it uses a script
  (Chinese/Japanese/Korean, Cyrillic, Arabic, Devanagari) that the reference's text does not. This
  is the "occasional Chinese character in English" problem.
- **JSON validity.** The 8 JSON prompts ask for a JSON object. We try to parse the first `{...}`
  block. Shown next to the reference's rate.
- **First divergence.** The median index of the first token where the quant's greedy text differs
  from the reference's. A difference alone is not damage. It is context.

## The probe set and contamination

Probe set v1 has 160 texts in 6 groups (general English, chat, code, math, multilingual, JSON),
about 80,000 scored positions for a typical tokenizer. Sources and licenses are in
`src/quantgap/probes/data/SOURCES.md`.

The models may have seen these texts in training. That does not bias the result. We measure the
distance between **two versions of the same model** on the same text. Both versions saw the text
equally. The probe set is a ruler, not an exam.

Only content tokens are scored. Special tokens (BOS, chat role markers) are masked out. In chat
rows only the assistant's reply is scored.

## What this does not tell you

- It is **not a skill benchmark**. A quant with more drift is not always worse at your task, and
  a quant with less drift is not always better.
- It is **not a speed benchmark.** Use `llama-bench` or llmfit for speed.
- A **fine-tune** is not a quant. If the quant was made from a different model than the pack, the
  drift includes the fine-tuning. quantgap warns when it can tell.
- Drift is measured on this probe set. A quant could be worse on a kind of text we do not cover.
