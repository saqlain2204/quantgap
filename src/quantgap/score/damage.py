"""Visible-damage checks on greedy generations.

Plain words: KLD can look small while a model still does something obviously wrong, like
repeating itself forever, dropping a Chinese character into English text, or breaking JSON.
We run short greedy generations and count those failures, always next to the reference's own.
"""

from __future__ import annotations

import json
from collections import Counter
from statistics import median

LOOP_NGRAM = 8
LOOP_REPEATS = 4
TAIL_WINDOW = 64
TAIL_MIN_DISTINCT = 12

# Unicode ranges for the scripts that most often leak into the wrong place.
_SCRIPTS: dict[str, list[tuple[int, int]]] = {
    "cjk": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x3040, 0x30FF), (0xAC00, 0xD7AF)],
    "cyrillic": [(0x0400, 0x052F)],
    "arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "devanagari": [(0x0900, 0x097F)],
}
# Groups where only the reference's own scripts are expected.
SCRIPT_CHECK_GROUPS = ("general", "code", "math", "json")


def is_loop(ids: list[int]) -> bool:
    """True if any 8-token run repeats 4+ times, or the last 64 tokens use fewer than 12 tokens."""
    n = len(ids)
    if n >= LOOP_NGRAM * LOOP_REPEATS:
        counts = Counter(tuple(ids[i : i + LOOP_NGRAM]) for i in range(n - LOOP_NGRAM + 1))
        if counts.most_common(1)[0][1] >= LOOP_REPEATS:
            return True
    return n >= TAIL_WINDOW and len(set(ids[-TAIL_WINDOW:])) < TAIL_MIN_DISTINCT


def scripts_in(text: str) -> set[str]:
    found: set[str] = set()
    for ch in text:
        c = ord(ch)
        if c < 0x0400:
            continue
        for name, ranges in _SCRIPTS.items():
            if name not in found and any(lo <= c <= hi for lo, hi in ranges):
                found.add(name)
    return found


def has_wrong_script(quant_text: str, ref_text: str) -> bool:
    """True if the quant text uses a script the reference text does not."""
    return bool(scripts_in(quant_text) - scripts_in(ref_text))


def first_json_block(text: str) -> str | None:
    """The first balanced {...} block in the text, or None."""
    start = text.find("{")
    if start < 0:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def json_valid(text: str) -> bool:
    block = first_json_block(text)
    if block is None:
        return False
    try:
        return isinstance(json.loads(block), dict)
    except ValueError:
        return False


def first_divergence(quant_ids: list[int], ref_ids: list[int]) -> int:
    """Index of the first token where the two outputs differ (the length if one is a prefix)."""
    for i, (a, b) in enumerate(zip(quant_ids, ref_ids, strict=False)):
        if a != b:
            return i
    return min(len(quant_ids), len(ref_ids))


def median_or_zero(values: list[int]) -> float:
    return float(median(values)) if values else 0.0
