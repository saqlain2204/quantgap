"""Shared fixtures. The tiny model is generated once per test session into a temp folder."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_make_tiny():
    spec = importlib.util.spec_from_file_location(
        "make_tiny_model", ROOT / "scripts" / "make_tiny_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MINI_PROBES = []


def _mini_rows():
    rows = []
    sid = 0

    def add(group, **kw):
        nonlocal sid
        row = {
            "seq_id": sid,
            "group": group,
            "kind": "text",
            "lang": "en",
            "text": None,
            "messages": None,
            "license": "CC0-1.0",
            "source": "test",
            "greedy": False,
            "greedy_prompt": None,
        }
        row.update(kw)
        rows.append(row)
        sid += 1

    texts = {
        "general": [
            "The quick brown fox jumps over the lazy dog. " * 6,
            "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen. " * 4,
            "A glacier is a persistent body of dense ice that moves under its own weight. " * 4,
            "Chess is a board game for two players. The pieces move in set ways. " * 5,
        ],
        "code": [
            "def add(a, b):\n    return a + b\n\nclass Thing:\n    pass\n" * 8,
            "for i in range(10):\n    print(i * i)\n" * 12,
            "function f(x) { return x * 2; }\nconsole.log(f(21));\n" * 8,
            "SELECT id, name FROM users WHERE age > 30 ORDER BY name;\n" * 8,
        ],
        "math": [
            "Question: Tom has 3 apples and buys 4 more. How many apples?\nThe answer is 7.\n" * 5,
            "Question: A box holds 12 eggs. How many in 5 boxes?\nThe answer is 60.\n" * 5,
            "Question: 9 plus 8 minus 3?\nThe answer is 14.\n" * 8,
            "Question: Sam walks 2 km a day for 6 days. Total?\nThe answer is 12.\n" * 6,
        ],
        "multilingual": [
            "El rápido zorro marrón salta sobre el perro perezoso. " * 6,
            "你好，世界。今天天气很好，我们一起去公园散步吧。" * 6,
            "नमस्ते दुनिया. आज मौसम बहुत अच्छा है। " * 6,
            "مرحبا بالعالم. الطقس جميل اليوم. " * 6,
        ],
    }
    for group, items in texts.items():
        for j, t in enumerate(items):
            add(group, text=t, greedy=j < 2, lang="en" if group != "multilingual" else "xx")
    for j in range(4):
        instr = "Return a JSON object with keys name and age. Output only the JSON.\n"
        obj = json.dumps({"name": f"P{j}", "age": 20 + j}, indent=2)
        add("json", text=instr + obj + "\n" + obj, greedy=True, greedy_prompt=instr)
    for j in range(4):
        add(
            "chat",
            kind="chat",
            messages=[
                {"role": "user", "content": f"Say something about the number {j}."},
                {"role": "assistant", "content": "The number is nice. Numbers are useful. " * 5},
            ],
            greedy=j < 2,
        )
    return rows


@pytest.fixture(scope="session")
def mini_probes_file(tmp_path_factory) -> Path:
    p = tmp_path_factory.mktemp("probes") / "probes_mini.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in _mini_rows():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return p


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("tiny_model")
    mod = _load_make_tiny()
    try:
        import gguf  # noqa: F401
        import llama_cpp  # noqa: F401

        with_gguf = True
    except ImportError:
        with_gguf = False
    mod.build(out, with_gguf=with_gguf)
    return out


@pytest.fixture(scope="session")
def tiny_hf(tiny_dir):
    pytest.importorskip("torch")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tiny_dir / "hf")
    model = AutoModelForCausalLM.from_pretrained(tiny_dir / "hf")
    model.eval()
    return model, tok


@pytest.fixture(scope="session")
def tiny_pack_dir(tiny_hf, mini_probes_file, tmp_path_factory) -> Path:
    from quantgap.pack.runner import build_pack_from_objects

    model, tok = tiny_hf
    out = tmp_path_factory.mktemp("tiny_pack")
    build_pack_from_objects(
        model,
        tok,
        out,
        repo="test/tiny-llama",
        probes_file=mini_probes_file,
        k=32,
        batch_size=4,
        max_new_tokens=48,
        dtype="float32",
        max_len=200,
    )
    return out


@pytest.fixture(scope="session")
def tiny_pack(tiny_pack_dir):
    from quantgap.pack import Pack

    return Pack.load(tiny_pack_dir)


@pytest.fixture(scope="session")
def tiny_gguf_f16(tiny_dir) -> Path:
    p = tiny_dir / "tiny-f16.gguf"
    if not p.exists():
        pytest.skip("llama-cpp-python / gguf not installed")
    return p


@pytest.fixture(scope="session")
def tiny_gguf_q4(tiny_dir) -> Path:
    p = tiny_dir / "tiny-q4_0.gguf"
    if not p.exists():
        pytest.skip("llama-cpp-python / gguf not installed")
    return p
