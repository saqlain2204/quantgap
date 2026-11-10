"""Top-k KLD with a tail bucket, top-1 agreement, perplexity ratio.

Plain words: at each position the reference model gives probabilities to its k best next tokens
and the rest of the probability ("the tail") is lumped into one bucket. We ask the quant for
its probabilities on the same k tokens, lump its leftover into a tail bucket too, and measure
how far apart the two sets of numbers are. Lumping can only make two distributions look
closer, so this number is a lower bound of the true KLD (see docs/method.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12  # floor for the quant's probabilities before taking a log
TAIL_SKIP = 1e-9  # ignore the reference tail term when its mass is this small


@dataclass
class KLResult:
    kl: np.ndarray  # (n,) per-position KL_k
    clamps: int  # how many probabilities had to be raised to EPS


def kld_topk(p_logp: np.ndarray, q_logp: np.ndarray) -> KLResult:
    """Per-position top-k + tail KLD.

    p_logp: (n, k) reference log probabilities of its own top-k ids.
    q_logp: (n, k) quant log probabilities of the *same* ids. NaN means "the backend could not
        tell" (a server that returns only its own top-N). Such ids are moved into the tail bucket
        for both distributions, which keeps the result a valid lower bound.
    """
    p_logp = np.asarray(p_logp, dtype=np.float64)
    q_logp = np.asarray(q_logp, dtype=np.float64)
    valid = ~np.isnan(q_logp)
    p_lp = np.where(valid, p_logp, 0.0)
    p = np.where(valid, np.exp(p_logp), 0.0)
    q = np.where(valid, np.exp(np.where(valid, q_logp, 0.0)), 0.0)

    clamped = valid & (q < EPS)
    q_safe = np.maximum(q, EPS)
    head = np.where(valid, p * (p_lp - np.log(q_safe)), 0.0).sum(axis=1)

    p_tail = np.clip(1.0 - p.sum(axis=1), 0.0, 1.0)
    q_tail_raw = 1.0 - q.sum(axis=1)
    q_tail = np.maximum(q_tail_raw, EPS)
    use_tail = p_tail >= TAIL_SKIP
    tail = np.where(use_tail, p_tail * (np.log(np.maximum(p_tail, EPS)) - np.log(q_tail)), 0.0)
    # A clearly negative tail means the quant's probabilities add up to more than 1: broken.
    broken_tail = use_tail & (q_tail_raw < -1e-6)
    clamps = int(clamped.sum() + broken_tail.sum())
    return KLResult(head + tail, clamps)


def log_softmax(x: np.ndarray) -> np.ndarray:
    """Stable log-softmax in float64 (subtract the max first)."""
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def kld_full(p_logits: np.ndarray, q_logits: np.ndarray) -> np.ndarray:
    """Exact KLD over the whole vocabulary (used by tests and `doctor --selftest`)."""
    p_lp = log_softmax(p_logits)
    q_lp = log_softmax(q_logits)
    return (np.exp(p_lp) * (p_lp - q_lp)).sum(axis=-1)


def ppl_ratio(logp_target_quant: np.ndarray, logp_target_ref: np.ndarray) -> float:
    """exp(mean(-log q)) / exp(mean(-log p)) over scored positions. 1.0 means no change."""
    return float(np.exp(np.mean(logp_target_ref - logp_target_quant)))
