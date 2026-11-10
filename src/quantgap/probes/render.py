"""Turn probe rows into token ids and a score mask, using the reference model's tokenizer.

Position `pos` predicts token `pos + 1`. A position is scored when token `pos + 1` is real
content: not a special token, and for chat rows, inside the assistant reply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quantgap.models import ProbeSeq

MAX_LEN = 512
GREEDY_PROMPT_TOKENS = 64
_BOUNDARIES = ("\n", ". ", "。", "।", "؟", "? ", "! ", "! ", "}", ";")
FALLBACK_TEMPLATE_NOTE = "no chat template on tokenizer: used plain 'User:/Assistant:' text"


@dataclass
class RenderedSeq:
    seq_id: int
    group: str
    token_ids: list[int]
    score_mask: list[bool]  # per token: is this token a scored target
    n_scored: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class RenderedGreedy:
    seq_id: int
    group: str
    prompt_ids: list[int]
    lang: str = "en"


def _encode(tok: Any, text: str) -> list[int]:
    return list(tok(text, add_special_tokens=True)["input_ids"])


def _truncate_text(tok: Any, text: str, max_len: int) -> str:
    """Longest prefix of `text` that fits in max_len tokens, cut at a line or sentence end."""
    if len(_encode(tok, text)) <= max_len:
        return text
    lo, hi = 1, len(text)
    while lo < hi:  # binary search on characters
        mid = (lo + hi + 1) // 2
        if len(_encode(tok, text[:mid])) <= max_len:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    best = max((cut.rfind(b) + len(b) for b in _BOUNDARIES if cut.rfind(b) > 0), default=0)
    if best >= int(len(cut) * 0.6):
        cut = cut[:best]
    return cut.rstrip()


def _special_ids(tok: Any) -> set[int]:
    return {int(i) for i in getattr(tok, "all_special_ids", [])}


def _ids_and_start(tok: Any, full: str, prefix: str, add_special: bool) -> tuple[list[int], int]:
    """Token ids of `full` and the index of the first token that lies entirely after `prefix`.

    A BPE token can straddle the boundary (for example " The" glues the space of the prompt to
    the first word of the reply). Such a token counts as context and is not scored.
    """
    try:
        enc = tok(full, add_special_tokens=add_special, return_offsets_mapping=True)
        ids = list(enc["input_ids"])
        offs = enc["offset_mapping"]
        start = next((i for i, (a, _) in enumerate(offs) if a >= len(prefix)), len(ids))
        return ids, start
    except (NotImplementedError, TypeError, ValueError):  # slow tokenizer: no offsets
        ids = list(tok(full, add_special_tokens=add_special)["input_ids"])
        plen = len(tok(prefix, add_special_tokens=add_special)["input_ids"])
        return ids, plen


def _chat_ids(tok: Any, messages: list[dict[str, str]]) -> tuple[list[int], int, list[str]]:
    """Return (ids, assistant_start_index, notes)."""
    notes: list[str] = []
    if getattr(tok, "chat_template", None):
        prefix = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
        full = tok.apply_chat_template(messages, tokenize=False)
        # The template already wrote any BOS token as text, so do not add special tokens again.
        ids, start = _ids_and_start(tok, full, prefix, add_special=False)
        if not full.startswith(prefix):
            notes.append("chat prefix not a prefix of full render: scored all non-special tokens")
            start = 0
        return ids, start, notes
    notes.append(FALLBACK_TEMPLATE_NOTE)
    prefix = _plain_chat_prefix(messages[:-1]) + "Assistant:"
    full = prefix + " " + messages[-1]["content"]
    ids, start = _ids_and_start(tok, full, prefix, add_special=True)
    return ids, start, notes


def _plain_chat_prefix(messages: list[dict[str, str]]) -> str:
    return "".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n" for m in messages
    )


def render_probes(
    probes: list[ProbeSeq], tok: Any, max_len: int = MAX_LEN
) -> tuple[list[RenderedSeq], list[RenderedGreedy]]:
    """Render every probe with `tok`. Deterministic: same inputs give identical ids."""
    special = _special_ids(tok)
    seqs: list[RenderedSeq] = []
    greedy: list[RenderedGreedy] = []
    for p in probes:
        notes: list[str] = []
        if p.kind == "chat":
            assert p.messages, f"probe {p.seq_id}: chat row without messages"
            msgs = p.messages
            ids, start, notes = _chat_ids(tok, msgs)
            if len(ids) > max_len:
                # Shorten the assistant reply, never the prompt.
                reply = msgs[-1]["content"]
                budget = max_len - start
                short = _truncate_text(tok, reply, max(8, budget - 8))
                msgs = msgs[:-1] + [{"role": msgs[-1]["role"], "content": short}]
                ids, start, notes = _chat_ids(tok, msgs)
                ids = ids[:max_len]
        else:
            assert p.text, f"probe {p.seq_id}: text row without text"
            ids = _encode(tok, _truncate_text(tok, p.text, max_len))
            start = 0
        mask = [False] * len(ids)
        for t in range(1, len(ids)):
            mask[t] = t >= start and ids[t] not in special
        seqs.append(
            RenderedSeq(p.seq_id, p.group, ids, mask, n_scored=int(np.sum(mask)), notes=notes)
        )
        if p.greedy:
            if p.kind == "chat":
                gm = [m for m in (p.messages or [])[:-1]]
                if getattr(tok, "chat_template", None):
                    s = tok.apply_chat_template(gm, add_generation_prompt=True, tokenize=False)
                    gids = list(tok(s, add_special_tokens=False)["input_ids"])
                else:
                    s = "".join(
                        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}\n"
                        for m in gm
                    )
                    gids = _encode(tok, s + "Assistant:")
            else:
                gtext = p.greedy_prompt or p.text or ""
                gids = _encode(tok, gtext)
                if p.greedy_prompt is None:
                    gids = gids[:GREEDY_PROMPT_TOKENS]
            greedy.append(RenderedGreedy(p.seq_id, p.group, gids, p.lang))
    return seqs, greedy
