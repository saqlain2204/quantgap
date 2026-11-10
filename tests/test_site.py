import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _site():
    spec = importlib.util.spec_from_file_location("qg_site_build", ROOT / "site" / "build.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qg_site_build"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(root, label, kld, approx=False, sha=None, pack="abcd1234abcd1234"):
    from .test_recommend import fake

    r = fake(label, 1.0 + kld, kld, pack=pack)
    r.pack_name = f"Tiny-Model@{pack}"
    r.quant.sha256 = sha or (label.encode().hex() * 8)[:64]
    r.approximate = approx
    d = root / "results" / pack
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{r.quant.sha256}.json").write_text(r.to_json(), encoding="utf-8")


def test_t14_site_build(tmp_path):
    site = _site()
    _write(tmp_path, "Q8_0", 0.002)
    _write(tmp_path, "Q4_K_M", 0.04)
    _write(tmp_path, "NVFP4", 0.09, approx=True)
    (tmp_path / "results" / "abcd1234abcd1234" / "broken.json").write_text(
        "{not json", encoding="utf-8"
    )
    out = tmp_path / "dist"
    data = site.build(tmp_path / "results", out)

    pages = sorted(p.name for p in out.glob("*.html"))
    assert pages == ["index.html", "model-abcd1234abcd1234.html"]
    model_html = (out / "model-abcd1234abcd1234.html").read_text(encoding="utf-8")
    assert model_html.count('class="approx"') >= 1 and "NVFP4" in model_html
    assert model_html.index("Q8_0") < model_html.index("Q4_K_M")  # sorted by KLD
    # no broken links between pages or to static files
    for page in out.glob("*.html"):
        for href in re.findall(r'(?:href|src)="([^"#:]+)"', page.read_text(encoding="utf-8")):
            assert (out / href).exists(), f"{page.name} links to missing {href}"
    # data.json validates against the schema
    raw = json.loads((out / "data.json").read_text(encoding="utf-8"))
    parsed = site.SiteData.model_validate(raw)
    assert len(parsed.packs) == 1 and len(parsed.packs[0].results) == 3
    assert sum(r.approximate for r in parsed.packs[0].results) == 1
    assert data.schema_ == "quantgap.site/1"


def test_empty_results_still_builds(tmp_path):
    site = _site()
    data = site.build(tmp_path / "none", tmp_path / "dist")
    assert data.packs == []
    assert "No results yet" in (tmp_path / "dist" / "index.html").read_text(encoding="utf-8")


def test_duplicates_keep_newest(tmp_path):
    site = _site()
    _write(tmp_path, "Q4_K_M", 0.04, sha="a" * 64)
    _write(tmp_path, "Q4_K_M", 0.05, sha="a" * 64)
    results, _ = site.load_results(tmp_path / "results")
    assert len(results) == 1
