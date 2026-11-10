import numpy as np

from quantgap.backends import BackendSettings, ResolvedSource
from quantgap.backends.hf import HFBackend
from quantgap.score.run import score_quant


def test_t15_selftest_reference_vs_own_pack(tiny_dir, tiny_pack):
    b = HFBackend()
    src = ResolvedSource(source="local", label="ref", path=tiny_dir / "hf", format="hf")
    b.load(src, BackendSettings(ctx=2048))
    res = score_quant(tiny_pack, b, src, BackendSettings(ctx=2048), do_damage=False)
    m = res.metrics
    assert m.kld_mean < 1e-6
    assert m.top1_agree == 1.0
    assert abs(m.ppl_ratio - 1.0) < 1e-4
    assert not res.approximate and not res.nondeterministic
    assert set(m.by_group) == {"general", "code", "math", "multilingual", "json", "chat"}


def test_selftest_damage_matches_reference(tiny_dir, tiny_pack):
    b = HFBackend()
    src = ResolvedSource(source="local", label="ref", path=tiny_dir / "hf", format="hf")
    b.load(src, BackendSettings())
    res = score_quant(tiny_pack, b, src, BackendSettings(), do_damage=True)
    d = res.metrics.damage
    assert d is not None
    assert d.loops == d.loops_ref  # same model, same greedy output
    assert d.first_divergence_median >= 20  # greedy outputs agree with the reference


def test_quick_mode_scores_fewer_positions(tiny_dir, tiny_pack):
    b = HFBackend()
    src = ResolvedSource(source="local", label="ref", path=tiny_dir / "hf", format="hf")
    b.load(src, BackendSettings())
    full = score_quant(tiny_pack, b, src, BackendSettings(), do_damage=False)
    quick = score_quant(tiny_pack, b, src, BackendSettings(), quick=True, do_damage=False)
    assert quick.quick and not quick.submittable
    assert quick.metrics.n_positions < full.metrics.n_positions
    assert np.isfinite(quick.metrics.kld_mean)
