import struct

import pytest

from quantgap.hub.resolve import (
    ResolveError,
    expand_glob,
    label_from_filename,
    parse_hf_url,
    read_gguf_metadata,
)
from quantgap.hub.submit import SubmitError, build_submission, submit

from .test_recommend import fake


def test_t12_parse_hf_url():
    r = parse_hf_url("hf://org/repo/sub/file.gguf")
    assert (r.repo, r.path) == ("org/repo", "sub/file.gguf")
    r = parse_hf_url("hf://org/repo")
    assert (r.repo, r.path) == ("org/repo", None)
    with pytest.raises(ResolveError, match="org/repo"):
        parse_hf_url("hf://justone")
    with pytest.raises(ResolveError):
        parse_hf_url("https://x/y")


def test_t12_glob_expands_with_mocked_list():
    files = [
        "a-Q4_K_M.gguf",
        "a-Q8_0.gguf",
        "sub/a-IQ4_XS.gguf",
        "README.md",
        "big-00002-of-00003.gguf",
        "big-00001-of-00003.gguf",
    ]
    assert expand_glob("o/r", "*Q4_K*", files) == ["a-Q4_K_M.gguf"]
    assert "sub/a-IQ4_XS.gguf" in expand_glob("o/r", "*.gguf", files)
    assert "big-00002-of-00003.gguf" not in expand_glob("o/r", "*.gguf", files)
    with pytest.raises(ResolveError, match="No files"):
        expand_glob("o/r", "*nothing*", files)


def test_labels():
    assert label_from_filename("Qwen3-27B-UD-Q4_K_XL.gguf") == "UD-Q4_K_XL"
    assert label_from_filename("m-IQ3_XXS.gguf") == "IQ3_XXS"
    assert label_from_filename("m-Q4_K_M.gguf") == "Q4_K_M"


def test_gguf_metadata_reader(tiny_gguf_f16):
    meta = read_gguf_metadata(tiny_gguf_f16, {"general.architecture", "llama.block_count"})
    assert meta["general.architecture"] == "llama"
    assert meta["llama.block_count"] == 2


def test_not_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"nope" + struct.pack("<I", 1))
    with pytest.raises(ResolveError):
        read_gguf_metadata(p)


def _result(**kw):
    r = fake("Q4_K_M", 1.0, 0.05, pack="abc123")
    r.quant.sha256 = "f" * 64
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def test_t13_submit_dry_run(tmp_path):
    f = tmp_path / "r.json"
    f.write_text(_result().to_json(), encoding="utf-8")
    sub, url = submit(f, "quantgap/results", dry_run=True)
    assert sub.path == f"results/abc123/{'f' * 64}.json"
    assert url is None
    assert "Q4_K_M" in sub.pr_body and "SHA-256" in sub.pr_body


def test_submit_refuses_quick_and_unhashed():
    with pytest.raises(SubmitError, match="quick"):
        build_submission(_result(quick=True))
    r = _result()
    r.quant.sha256 = None
    with pytest.raises(SubmitError, match="hash"):
        build_submission(r)


def _manifest(repo):
    from quantgap.models import (
        GreedyInfo,
        Manifest,
        ModelRef,
        ProbesInfo,
        RunnerInfo,
        TokenizerInfo,
    )

    return Manifest(
        model=ModelRef(repo=repo),
        tokenizer=TokenizerInfo(vocab_size=10),
        probes=ProbesInfo(set="v1", sha256="x", n_sequences=1, n_positions=1),
        k=4,
        greedy=GreedyInfo(n_prompts=1, max_new_tokens=1),
        runner=RunnerInfo(quantgap="0.1.0"),
        created_at="now",
    )


def test_auto_find_pack_exact_then_family(monkeypatch):
    from quantgap.config import Settings
    from quantgap.hub import packs as hp

    monkeypatch.setattr(hp, "load_pack", lambda ref: ref)
    packs = [
        ("p/inst", _manifest("Qwen/Qwen2.5-0.5B-Instruct")),
        ("p/other", _manifest("meta/Llama-3-8B")),
    ]
    monkeypatch.setattr(hp, "list_packs", lambda settings: packs)
    # a GGUF that names the parent model still finds the instruct pack
    assert hp.auto_find_pack("Qwen/Qwen2.5-0.5B", Settings()) == "p/inst"
    # exact match wins over a family match
    packs.append(("p/base", _manifest("Qwen/Qwen2.5-0.5B")))
    assert hp.auto_find_pack("Qwen/Qwen2.5-0.5B", Settings()) == "p/base"
    with pytest.raises(hp.PackNotFound, match="No pack found"):
        hp.auto_find_pack("mistralai/Mistral-7B", Settings())
    with pytest.raises(hp.PackNotFound, match="does not say"):
        hp.auto_find_pack(None, Settings())


def test_dataset_card_declares_one_config_per_parquet_kind():
    """The Hub viewer fails if differently shaped parquet files share one config."""
    import yaml

    from quantgap.hub.packs import dataset_card

    front = dataset_card(_manifest("a/b")).split("---")[1]
    cfg = {c["config_name"]: c["data_files"] for c in yaml.safe_load(front)["configs"]}
    assert cfg == {"topk": "topk-*.parquet", "probes": "probes.parquet", "greedy": "greedy.parquet"}
