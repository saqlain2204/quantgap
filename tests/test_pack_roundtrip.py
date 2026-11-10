import json
import shutil

import numpy as np
import pytest

from quantgap.pack import Pack, PackIntegrityError
from quantgap.probes import load_probes, render_probes


def test_t1_render_is_deterministic(tiny_hf, mini_probes_file):
    _, tok = tiny_hf
    probes = load_probes(path=mini_probes_file)
    a, ga = render_probes(probes, tok, max_len=200)
    b, gb = render_probes(probes, tok, max_len=200)
    assert [s.token_ids for s in a] == [s.token_ids for s in b]
    assert [s.score_mask for s in a] == [s.score_mask for s in b]
    assert [g.prompt_ids for g in ga] == [g.prompt_ids for g in gb]
    assert all(len(s.token_ids) <= 200 for s in a)


def test_chat_mask_only_scores_assistant_reply(tiny_hf, mini_probes_file):
    _, tok = tiny_hf
    probes = [p for p in load_probes(path=mini_probes_file) if p.kind == "chat"]
    seqs, _ = render_probes(probes, tok, max_len=200)
    s = seqs[0]
    first_scored = s.score_mask.index(True)
    prompt = tok.apply_chat_template(probes[0].messages[:-1], add_generation_prompt=True, tokenize=False)
    assert first_scored >= len(tok(prompt, add_special_tokens=False)["input_ids"])


def test_t4_pack_roundtrip(tiny_pack_dir, tmp_path):
    pack = Pack.load(tiny_pack_dir)
    m = pack.manifest
    assert m.k == 32 and m.probes.n_positions > 100
    topk = pack.topk()
    assert sum(len(t.pos) for t in topk.values()) == m.probes.n_positions
    t = next(iter(topk.values()))
    assert t.ids.shape == (len(t.pos), 32)
    assert (np.diff(t.logp, axis=1) <= 1e-6).all()  # sorted highest first
    # the stored target logp matches the top-1 when the target is the argmax
    hit = t.ids[:, 0] == t.target
    assert np.allclose(t.logp[hit, 0], t.logp_target[hit], atol=1e-5)
    # manifest hash is stable and pack id is 16 hex chars
    assert pack.pack_id == Pack.load(tiny_pack_dir).pack_id
    assert len(pack.pack_id) == 16
    # tamper with one byte
    copy = tmp_path / "copy"
    shutil.copytree(tiny_pack_dir, copy)
    shard = copy / "topk-00000.parquet"
    data = bytearray(shard.read_bytes())
    data[len(data) // 2] ^= 0xFF
    shard.write_bytes(bytes(data))
    with pytest.raises(PackIntegrityError, match="topk-00000"):
        Pack.load(copy)


def test_manifest_is_valid_json_with_schema(tiny_pack_dir):
    data = json.loads((tiny_pack_dir / "manifest.json").read_text(encoding="utf-8"))
    assert data["schema"] == "quantgap.pack/1"
    assert data["tokenizer"]["check_ids"]
    assert set(data["files"]) >= {"probes.parquet", "greedy.parquet", "topk-00000.parquet"}


def test_greedy_rows_saved(tiny_pack):
    rows = tiny_pack.greedy()
    assert len(rows) == tiny_pack.manifest.greedy.n_prompts > 0
    assert all(len(r.prompt_ids) > 0 for r in rows)
