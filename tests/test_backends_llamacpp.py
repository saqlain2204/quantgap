import pytest

pytest.importorskip("llama_cpp")

from quantgap.backends import BackendSettings, ResolvedSource  # noqa: E402
from quantgap.backends.llamacpp import LlamaCppBackend  # noqa: E402
from quantgap.score.run import ScoreError, score_quant  # noqa: E402


def _score(pack, path, **kw):
    b = LlamaCppBackend()
    s = BackendSettings(ctx=kw.pop("ctx", 2048), n_gpu_layers=0)
    src = ResolvedSource(source=str(path), label=path.stem, path=path, size_bytes=path.stat().st_size)
    b.load(src, s)
    try:
        return score_quant(pack, b, src, s, **kw)
    finally:
        b.close()


def test_f16_gguf_matches_pack(tiny_pack, tiny_gguf_f16):
    r = _score(tiny_pack, tiny_gguf_f16, do_damage=False)
    assert r.metrics.kld_mean < 1e-3
    assert r.metrics.top1_agree > 0.97


def test_t8_q4_is_worse_than_f16(tiny_pack, tiny_gguf_f16, tiny_gguf_q4):
    f16 = _score(tiny_pack, tiny_gguf_f16, do_damage=False)
    q4 = _score(tiny_pack, tiny_gguf_q4, do_damage=False)
    assert f16.metrics.kld_mean < q4.metrics.kld_mean
    assert f16.metrics.top1_agree >= q4.metrics.top1_agree
    assert q4.metrics.kld_mean > 10 * f16.metrics.kld_mean


def test_gguf_damage_checks_run(tiny_pack, tiny_gguf_f16):
    r = _score(tiny_pack, tiny_gguf_f16, do_damage=True, quick=True)
    assert r.metrics.damage is not None and r.metrics.damage.n_prompts > 0


def test_t16_ctx_too_small_refuses(tiny_pack, tiny_gguf_f16):
    with pytest.raises(ScoreError, match=r"--ctx \d+ or more"):
        _score(tiny_pack, tiny_gguf_f16, ctx=64, do_damage=False)
