"""Tokenizer compatibility check.

A quant can only be compared to a pack if both use the same token ids. We check this before
scoring and stop with a clear error if it fails. We never "fix" a mismatch silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quantgap.models import TokenizerInfo

# Mixed scripts, code, numbers, emoji, odd spacing. Special tokens are checked by id, not by text.
CHECK_STRING = (
    "Hello, world! 你好，世界。 नमस्ते दुनिया. مرحبا بالعالم. ¿Qué tal?\n"
    "def f(x):\n    return {'a': [1, 2.5e-3, None]}  # 😀🚀\n"
    "  two  spaces\ttab 12345 3.14159 https://example.com/a?b=c&d=e\n"
)


class TokenizerMismatch(RuntimeError):
    """The quant's tokenizer is not the same as the pack's."""


@dataclass
class TokFingerprint:
    vocab_size: int
    bos_id: int | None
    eos_id: int | None
    check_ids: list[int] = field(default_factory=list)


def check_tokenizer(pack_tok: TokenizerInfo, quant: TokFingerprint, pack_top_ids_max: int) -> None:
    """Raise TokenizerMismatch if the quant cannot be compared to the pack.

    `pack_top_ids_max` is the largest token id seen anywhere in the pack's top-k lists and targets.
    A quant may have extra padding ids beyond the reference vocabulary if those never appear there.
    """
    # 1. vocabulary size
    # A server cannot tell us its vocabulary size (vocab_size <= 0): skip this one check.
    if quant.vocab_size > 0 and quant.vocab_size != pack_tok.vocab_size:
        if quant.vocab_size < pack_tok.vocab_size:
            raise TokenizerMismatch(
                f"Check 1 (vocab size) failed: quant has {quant.vocab_size} tokens, "
                f"pack expects {pack_tok.vocab_size}."
            )
        if pack_top_ids_max >= pack_tok.vocab_size:  # pragma: no cover - defensive
            raise TokenizerMismatch(
                "Check 1 (vocab size) failed: pack ids exceed its own vocabulary."
            )
        # extra padding ids are fine: no pack id points at them.
    # 2. fixed check string
    if pack_tok.check_ids and quant.check_ids != pack_tok.check_ids:
        a, b = pack_tok.check_ids, quant.check_ids
        i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        got = b[i] if i < len(b) else "(end)"
        want = a[i] if i < len(a) else "(end)"
        raise TokenizerMismatch(
            f"Check 2 (encode a fixed test string) failed at token index {i}: "
            f"pack has id {want}, quant has id {got}. "
            "These models do not use the same tokenizer, so scores would be meaningless."
        )
    # 3. BOS / EOS ids
    for label, want, got in (
        ("BOS", pack_tok.bos_id, quant.bos_id),
        ("EOS", pack_tok.eos_id, quant.eos_id),
    ):
        if want is not None and got is not None and want != got:
            raise TokenizerMismatch(
                f"Check 3 ({label} id) failed: pack has {want}, quant has {got}."
            )
