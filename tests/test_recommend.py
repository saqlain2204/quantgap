import pytest

from quantgap.models import BackendInfo, Metrics, QuantInfo, ScoreResult
from quantgap.score.compare import MixedPacksError, build_report
from quantgap.score.recommend import recommend
from quantgap.util import GIB, parse_size


def fake(label, size_gb, kld, ci=None, pack="p1"):
    ci = ci or (kld * 0.9, kld * 1.1)
    return ScoreResult(
        pack_id=pack,
        quant=QuantInfo(source=label, size_bytes=int(size_gb * 1e9), label=label),
        backend=BackendInfo(name="x", version="1"),
        metrics=Metrics(
            kld_mean=kld,
            kld_mean_ci95=ci,
            kld_p99=0,
            kld_p999=0,
            top1_agree=0.9,
            ppl_ratio=1.0,
            n_positions=10,
            n_sequences=2,
        ),
        hardware="t",
        quantgap="0.1.0",
        created_at="now",
    )


def test_t11_budget_and_ties():
    rs = [fake("a", 0.6, 0.30), fake("b", 0.9, 0.10), fake("c", 1.2, 0.01)]
    # 1 GB cannot hold anything once the 1 GiB overhead is added
    assert recommend(rs, parse_size("1GB")).chosen is None
    budget = int(0.9e9 * 1.1 + GIB) + 10
    rec = recommend(rs, budget)
    assert rec.chosen.quant.label == "b"  # c does not fit; b has the lowest KLD that fits
    # ties: overlapping intervals prefer the smaller file
    rs2 = [fake("small", 0.6, 0.100, (0.08, 0.12)), fake("large", 0.9, 0.095, (0.07, 0.11))]
    rec2 = recommend(rs2, int(1e10))
    assert rec2.chosen.quant.label == "small"
    assert "not clearly different" in rec2.sentence


def test_clear_winner_beats_smaller_file():
    rs = [fake("small", 0.6, 0.30, (0.28, 0.32)), fake("large", 0.9, 0.05, (0.04, 0.06))]
    assert recommend(rs, int(1e10)).chosen.quant.label == "large"


def test_damage_warning():
    r = fake("x", 1, 0.1)
    from quantgap.models import DamageMetrics

    r.metrics.damage = DamageMetrics(
        n_prompts=48, loops=6, loops_ref=1, loop_rate=0.12, wrong_script=0, wrong_script_ref=0,
        wrong_script_rate=0, json_valid=8, json_valid_ref=8, json_n=8, json_valid_rate=1.0,
        first_divergence_median=10,
    )  # fmt: skip
    rec = recommend([r], int(1e10))
    assert rec.warnings and "loops" in rec.warnings[0]


def test_parse_size():
    assert parse_size("16GB") == 16_000_000_000
    assert parse_size("16GiB") == 16 * GIB
    assert parse_size("16000MB") == 16_000_000_000
    with pytest.raises(ValueError):
        parse_size("lots")


def test_t17_mixed_packs_refused():
    with pytest.raises(MixedPacksError, match="different packs"):
        build_report([fake("a", 1, 0.1, pack="p1"), fake("b", 1, 0.1, pack="p2")])
