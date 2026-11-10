import json
import re
import xml.dom.minidom as minidom

import pytest
from typer.testing import CliRunner

from quantgap.cli import app

runner = CliRunner()


def plain(r):
    """Output without colour codes and with line wrapping flattened (CI may force colour)."""
    return " ".join(re.sub(r"\[[0-9;]*m", "", r.output).split())


@pytest.mark.parametrize(
    "cmd",
    [[], ["pack"], ["packs"], ["packs", "list"], ["packs", "show"], ["score"], ["compare"],
     ["submit"], ["inspect"], ["doctor"]],
)  # fmt: skip
def test_t18_every_command_help_exits_zero(cmd):
    r = runner.invoke(app, [*cmd, "--help"])
    assert r.exit_code == 0, r.output
    assert "Usage" in plain(r)


def test_t18_doctor_runs_and_names_extras():
    r = runner.invoke(app, ["doctor"])
    assert r.exit_code == 0
    out = plain(r)
    # The extras must be printed literally (Rich would swallow "[llamacpp]" as markup).
    assert 'pip install "quantgap[llamacpp]"' in out or "ok llamacpp" in out


def test_error_messages_keep_square_brackets(monkeypatch):
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "torch" else real(name, *a, **k),
    )
    r = runner.invoke(app, ["pack", "x/y"])
    assert r.exit_code == 1
    assert 'pip install "quantgap[pack]"' in plain(r)


def test_version():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and "quantgap" in plain(r)


def test_score_unknown_file_is_a_clear_error():
    r = runner.invoke(app, ["score", "nope.gguf", "--pack", "x"])
    assert r.exit_code == 1
    assert "Cannot find" in plain(r)


def test_score_ollama_name_gets_a_hint():
    r = runner.invoke(app, ["score", "llama3:8b", "--pack", "x"])
    assert r.exit_code == 1 and "ollama show --modelfile" in plain(r)


def test_score_bad_backend():
    r = runner.invoke(app, ["score", "x", "--backend", "banana"])
    assert r.exit_code == 1 and "Unknown backend" in plain(r)


def test_score_and_compare_end_to_end(
    tiny_gguf_f16, tiny_gguf_q4, tiny_pack_dir, tmp_path, monkeypatch
):
    pytest.importorskip("llama_cpp")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "f16.json"
    r = runner.invoke(
        app,
        ["score", str(tiny_gguf_f16), "--pack", str(tiny_pack_dir), "--n-gpu-layers", "0",
         "--quick", "--json", str(out)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "KLD mean" in plain(r) and "quick" in plain(r)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["schema"] == "quantgap.result/1" and data["quick"] is True
    assert data["quant"]["sha256"] and data["quant"]["format"] == "gguf"

    # inspect and a refused submit of a quick result
    assert runner.invoke(app, ["inspect", str(out)]).exit_code == 0
    r = runner.invoke(app, ["submit", str(out), "--dry-run"])
    assert r.exit_code == 1 and "quick" in plain(r)

    chart = tmp_path / "chart.svg"
    r = runner.invoke(
        app,
        ["compare", str(tiny_gguf_f16), str(tiny_gguf_q4), "--pack", str(tiny_pack_dir),
         "--n-gpu-layers", "0", "--quick", "--no-damage", "--budget", "5GB", "--chart", str(chart)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "For 5.0 GB, use" in plain(r)
    minidom.parse(str(chart))  # valid SVG/XML
    assert (tmp_path / "quantgap-results").is_dir()


def test_selftest_flag_exists():
    assert "--selftest" in plain(runner.invoke(app, ["doctor", "--help"]))
