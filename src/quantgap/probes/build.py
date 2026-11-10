"""Build probe set v1 from open sources. Run by maintainers, never by users.

    python -m quantgap.probes.build --out src/quantgap/probes/data/probes_v1.jsonl

The output file is frozen and checked in. This script is only here so anyone can see where the
texts came from (see SOURCES.md next to the data file). Re-running it later may give different
text because Wikipedia changes. The checked-in file is the source of truth, not this script.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import httpx

UA = {"User-Agent": "quantgap-probe-builder/0.1 (open-source LLM quant tool)"}

GROUP_SIZES = {
    "general": 30,
    "chat": 30,
    "code": 28,
    "math": 28,
    "multilingual": 32,
    "json": 12,
}
GREEDY_PER_GROUP = 8

EN_TITLES = [
    "Photosynthesis", "Roman Empire", "Black hole", "Printing press", "Coral reef",
    "Industrial Revolution", "Plate tectonics", "Jazz", "Marathon", "Honey bee",
    "Great Wall of China", "Internet", "Volcano", "Silk Road", "Theory of relativity",
    "Penicillin", "Mount Everest", "Renaissance", "Chess", "Antarctica",
    "Electric car", "Tea", "Amazon rainforest", "Solar System", "DNA",
    "Bicycle", "Olympic Games", "Glacier", "Mathematics", "Moon",
    "Cheese", "Earthquake", "Telescope", "Bread", "Rainbow",
    "Library", "Bridge", "Salt", "Ocean", "Clock",
]  # fmt: skip
LANG_CHARS = {"es": 2200, "hi": 1300, "zh": 750, "ar": 1300}
LANG_COUNT = {"hi": 8, "zh": 8, "es": 8, "ar": 8}


def _get_json(client: httpx.Client, url: str, params: dict) -> dict:
    for attempt in range(4):
        r = client.get(url, params=params, headers=UA, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 * (attempt + 1))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"giving up on {url} {params}")


def _wiki_text(client: httpx.Client, lang: str, title: str) -> tuple[str, str] | None:
    data = _get_json(
        client,
        f"https://{lang}.wikipedia.org/w/api.php",
        {
            "action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
            "exsectionformat": "plain", "titles": title, "redirects": 1,
        },
    )  # fmt: skip
    page = next(iter(data["query"]["pages"].values()))
    text = page.get("extract")
    if not text:
        return None
    return page["title"], text


def _langlink(client: httpx.Client, title: str, lang: str) -> str | None:
    data = _get_json(
        client,
        "https://en.wikipedia.org/w/api.php",
        {
            "action": "query", "format": "json", "prop": "langlinks", "titles": title,
            "lllang": lang, "redirects": 1,
        },
    )  # fmt: skip
    page = next(iter(data["query"]["pages"].values()))
    links = page.get("langlinks") or []
    return links[0]["*"] if links else None


def _clip(text: str, n: int) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= n:
        return text
    cut = text[:n]
    para = cut.rfind("\n")
    if para > n * 0.6:
        return cut[:para].rstrip()
    for end in (". ", "。", "।", "؟ ", ". "):
        i = cut.rfind(end)
        if i > n * 0.6:
            return cut[: i + len(end)].rstrip()
    return cut.rstrip()


def build_general(client) -> list[dict]:
    rows = []
    for title in EN_TITLES:
        got = _wiki_text(client, "en", title)
        if not got:
            continue
        t, text = got
        rows.append(
            dict(group="general", kind="text", lang="en", text=_clip(text, 2200),
                 license="CC BY-SA 4.0", source=f"wikipedia:en:{t}")
        )  # fmt: skip
        if len(rows) == GROUP_SIZES["general"]:
            break
    return rows


def build_multilingual(client) -> list[dict]:
    rows = []
    for li, lang in enumerate(["hi", "zh", "es", "ar"]):
        pool = EN_TITLES[li * 7 :] + EN_TITLES[: li * 7]
        got_n = 0
        for en_title in pool:
            if got_n == LANG_COUNT[lang]:
                break
            local = _langlink(client, en_title, lang)
            if not local:
                continue
            got = _wiki_text(client, lang, local)
            if not got or len(got[1]) < 400:
                continue
            t, text = got
            rows.append(
                dict(group="multilingual", kind="text", lang=lang,
                     text=_clip(text, LANG_CHARS[lang]), license="CC BY-SA 4.0",
                     source=f"wikipedia:{lang}:{t}")
            )  # fmt: skip
            got_n += 1
    return rows


def build_math(client) -> list[dict]:
    data = _get_json(
        client,
        "https://datasets-server.huggingface.co/rows",
        {"dataset": "openai/gsm8k", "config": "main", "split": "train", "offset": 0, "length": 100},
    )
    rows = []
    for item in data["rows"]:
        r = item["row"]
        answer = re.sub(r"<<[^>]*>>", "", r["answer"])
        answer = answer.replace("####", "The answer is")
        text = f"Question: {r['question']}\nLet's think step by step.\n{answer}"
        rows.append(
            dict(group="math", kind="text", lang="en", text=text, license="MIT",
                 source=f"gsm8k:train:{item['row_idx']}")
        )  # fmt: skip
        if len(rows) == GROUP_SIZES["math"]:
            break
    return rows


def build_chat(client) -> list[dict]:
    data = _get_json(
        client,
        "https://datasets-server.huggingface.co/rows",
        {"dataset": "databricks/databricks-dolly-15k", "config": "default", "split": "train",
         "offset": 0, "length": 100},
    )  # fmt: skip
    rows = []
    for item in data["rows"]:
        r = item["row"]
        if r.get("context"):
            continue
        if not (250 <= len(r["response"]) <= 1500) or len(r["instruction"]) > 400:
            continue
        rows.append(
            dict(group="chat", kind="chat", lang="en",
                 messages=[{"role": "user", "content": r["instruction"].strip()},
                           {"role": "assistant", "content": r["response"].strip()}],
                 license="CC BY-SA 3.0", source=f"dolly-15k:{item['row_idx']}")
        )  # fmt: skip
        if len(rows) == GROUP_SIZES["chat"]:
            break
    return rows


# Hand-written snippets in other languages (own work, Apache-2.0).
OTHER_CODE = [
    ("javascript", """// Debounce: run `fn` only after `ms` milliseconds of silence.
function debounce(fn, ms) {
  let timer = null;
  return function (...args) {
    clearTimeout(timer);
    timer = setTimeout(() => fn.apply(this, args), ms);
  };
}

const onResize = debounce(() => {
  console.log("window size", window.innerWidth, window.innerHeight);
}, 200);
window.addEventListener("resize", onResize);
"""),
    ("c", """#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node *next;
} Node;

Node *push(Node *head, int value) {
    Node *n = malloc(sizeof *n);
    if (!n) return head;
    n->value = value;
    n->next = head;
    return n;
}

int main(void) {
    Node *list = NULL;
    for (int i = 0; i < 5; i++) list = push(list, i * i);
    for (Node *p = list; p; p = p->next) printf("%d\\n", p->value);
    while (list) { Node *next = list->next; free(list); list = next; }
    return 0;
}
"""),
    ("rust", """use std::collections::HashMap;

fn word_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for word in text.split_whitespace() {
        let key = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
        if !key.is_empty() {
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    counts
}

fn main() {
    let counts = word_counts("the quick brown fox jumps over the lazy dog the end");
    let mut pairs: Vec<_> = counts.into_iter().collect();
    pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (word, n) in pairs.iter().take(3) {
        println!("{word}: {n}");
    }
}
"""),
    ("sql", """-- Top 5 customers by total spend in 2025, with their order counts.
SELECT c.id,
       c.name,
       COUNT(o.id)        AS orders,
       SUM(o.total_cents) / 100.0 AS spend
FROM customers AS c
JOIN orders AS o ON o.customer_id = c.id
WHERE o.created_at >= '2025-01-01'
  AND o.created_at <  '2026-01-01'
  AND o.status = 'paid'
GROUP BY c.id, c.name
HAVING SUM(o.total_cents) > 0
ORDER BY spend DESC
LIMIT 5;
"""),
    ("go", """package main

import (
	"fmt"
	"sync"
)

func main() {
	var wg sync.WaitGroup
	results := make(chan int, 10)
	for i := 1; i <= 10; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			results <- n * n
		}(i)
	}
	wg.Wait()
	close(results)
	sum := 0
	for r := range results {
		sum += r
	}
	fmt.Println("sum of squares:", sum)
}
"""),
    ("bash", """#!/usr/bin/env bash
set -euo pipefail

# Keep only the newest N backups in a folder.
dir="${1:?usage: prune.sh DIR [KEEP]}"
keep="${2:-7}"

mapfile -t files < <(ls -1t "$dir"/*.tar.gz 2>/dev/null || true)
if (( ${#files[@]} > keep )); then
  for f in "${files[@]:keep}"; do
    echo "removing $f"
    rm -- "$f"
  done
fi
"""),
]


def _python_chunks(src_root: Path, n: int) -> list[tuple[str, str]]:
    """Chunks of this project's own Python code (Apache-2.0), cut at blank lines."""
    skip = {"probes", "__pycache__"}
    files = sorted(
        p for p in src_root.rglob("*.py")
        if not (set(p.relative_to(src_root).parts) & skip) and p.stat().st_size > 1500
    )  # fmt: skip
    chunks: list[tuple[str, str]] = []
    for p in files:
        text = p.read_text(encoding="utf-8")
        blocks = text.split("\n\n\n")
        cur = ""
        for b in blocks:
            if len(cur) + len(b) > 1800 and len(cur) > 600:
                chunks.append((f"quantgap:{p.relative_to(src_root).as_posix()}", cur.strip() + "\n"))
                cur = ""
            cur += b + "\n\n\n"
        if len(cur) > 600:
            chunks.append((f"quantgap:{p.relative_to(src_root).as_posix()}", cur.strip() + "\n"))
    # Spread picks across files so the group is varied.
    rng = random.Random(0)
    rng.shuffle(chunks)
    return chunks[:n]


def build_code(src_root: Path) -> list[dict]:
    rows = []
    for lang, text in OTHER_CODE:
        rows.append(
            dict(group="code", kind="text", lang=lang, text=text, license="Apache-2.0",
                 source=f"handwritten:{lang}")
        )  # fmt: skip
    for src, text in _python_chunks(src_root, GROUP_SIZES["code"] - len(OTHER_CODE)):
        rows.append(
            dict(group="code", kind="text", lang="python", text=text, license="Apache-2.0",
                 source=src)
        )  # fmt: skip
    return rows


_NAMES = ["Asha", "Mateo", "Li Wei", "Fatima", "Noah", "Priya", "Lucas", "Amara", "Hiro", "Sofia",
          "Omar", "Ingrid"]  # fmt: skip
_CITIES = ["Pune", "Lagos", "Oslo", "Lima", "Osaka", "Cairo", "Porto", "Quito", "Hanoi", "Leeds",
           "Accra", "Turin"]  # fmt: skip
_ITEMS = ["notebook", "kettle", "backpack", "lamp", "headphones", "water bottle", "desk fan",
          "keyboard", "umbrella", "camera strap", "tea set", "yoga mat"]  # fmt: skip


def build_json() -> list[dict]:
    rng = random.Random(7)
    rows = []
    for i in range(GROUP_SIZES["json"]):
        name, city, item = _NAMES[i], _CITIES[i], _ITEMS[i]
        kind = i % 4
        if kind == 0:
            instr = (f"Return a JSON object describing the person {name} who lives in {city}. "
                     "Use the keys name, city, age and hobbies. Output only the JSON.")
            obj = {"name": name, "city": city, "age": rng.randint(19, 64),
                   "hobbies": rng.sample(["chess", "cycling", "cooking", "painting", "hiking"], 2)}
        elif kind == 1:
            instr = (f"Return a JSON object for an order of one {item} shipped to {city}. "
                     "Use the keys order_id, item, quantity, price and shipping. Output only the JSON.")
            obj = {"order_id": f"A-{rng.randint(1000, 9999)}", "item": item,
                   "quantity": rng.randint(1, 4), "price": round(rng.uniform(5, 90), 2),
                   "shipping": {"city": city, "express": bool(rng.getrandbits(1))}}
        elif kind == 2:
            instr = (f"Return a JSON object with a list of three tasks for {name}. Each task has "
                     "the keys title, done and priority. Output only the JSON.")
            obj = {"owner": name, "tasks": [
                {"title": t, "done": bool(rng.getrandbits(1)), "priority": rng.randint(1, 3)}
                for t in rng.sample(["write report", "book tickets", "call plumber", "buy milk",
                                     "fix bike", "renew passport"], 3)]}
        else:
            instr = (f"Return a JSON object for the weather in {city} with the keys city, "
                     "temperature_c, conditions and wind. wind has speed_kmh and direction. "
                     "Output only the JSON.")
            obj = {"city": city, "temperature_c": rng.randint(-5, 38),
                   "conditions": rng.choice(["sunny", "cloudy", "rain", "fog"]),
                   "wind": {"speed_kmh": rng.randint(0, 40),
                            "direction": rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])}}
        text = instr + "\n" + json.dumps(obj, indent=2)
        rows.append(
            dict(group="json", kind="text", lang="en", text=text, license="CC0-1.0",
                 source="generated:build.py", greedy_prompt=instr + "\n")
        )  # fmt: skip
    return rows


def build(src_root: Path) -> list[dict]:
    with httpx.Client(follow_redirects=True) as client:
        groups = {
            "general": build_general(client),
            "chat": build_chat(client),
            "code": build_code(src_root),
            "math": build_math(client),
            "multilingual": build_multilingual(client),
            "json": build_json(),
        }
    rows: list[dict] = []
    for g, items in groups.items():
        for j, row in enumerate(items):
            row["greedy"] = j < GREEDY_PER_GROUP
            row.setdefault("greedy_prompt", None)
            row.setdefault("text", None)
            row.setdefault("messages", None)
            row["seq_id"] = len(rows)
            rows.append(row)
        print(f"{g}: {len(items)} rows")
    return rows


def write(rows: list[dict], out: Path) -> None:
    keys = ["seq_id", "group", "kind", "lang", "text", "messages", "license", "source", "greedy",
            "greedy_prompt"]  # fmt: skip
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keys}, ensure_ascii=False) + "\n")
    # SOURCES.md table
    lines = ["# Probe set v1 sources", "",
             "Every row of `probes_v1.jsonl` records its license and source. Summary:", "",
             "| Group | Source | License | Rows |", "|---|---|---|---|"]  # fmt: skip
    summary: dict[tuple[str, str, str], int] = {}
    for r in rows:
        src = r["source"].split(":")[0]
        summary[(r["group"], src, r["license"])] = summary.get((r["group"], src, r["license"]), 0) + 1
    for (g, s, lic), n in summary.items():
        lines.append(f"| {g} | {s} | {lic} | {n} |")
    lines += ["", "Wikipedia text is CC BY-SA 4.0 (author credit: the article editors, see the page "
              "history of each article named in `source`). Dolly-15k is CC BY-SA 3.0. GSM8K is MIT. "
              "Code and JSON rows are Apache-2.0 / CC0 (written for this project)."]  # fmt: skip
    (out.parent / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="src/quantgap/probes/data/probes_v1.jsonl")
    ap.add_argument("--src-root", default="src/quantgap")
    a = ap.parse_args()
    rows = build(Path(a.src_root))
    write(rows, Path(a.out))
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
