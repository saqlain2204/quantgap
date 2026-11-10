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
