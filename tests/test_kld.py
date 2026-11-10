import numpy as np

from quantgap.score.kld import kld_full, kld_topk, log_softmax, ppl_ratio
from quantgap.score.stats import bootstrap_mean_ci


def _topk(logits, k):
    lp = log_softmax(logits)
    idx = np.argsort(-lp, axis=-1)[:, :k]
    return idx, np.take_along_axis(lp, idx, axis=1), lp


def test_t2_kld_of_itself_is_zero():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(200, 512)) * 3
    idx, p_lp, lp = _topk(logits, 32)
    res = kld_topk(p_lp, np.take_along_axis(lp, idx, axis=1))
    assert np.max(np.abs(res.kl)) < 1e-12
    assert res.clamps == 0


def test_t3_topk_is_lower_bound_and_gap_shrinks():
    rng = np.random.default_rng(2)
    gaps = {8: [], 32: [], 128: []}
    for _ in range(1000):
        p_logits = rng.normal(size=(1, 512)) * 2
        q_logits = p_logits + rng.normal(size=(1, 512)) * 0.5
        exact = kld_full(p_logits, q_logits)[0]
        idx, p_lp, _ = _topk(p_logits, 128)
        q_lp = log_softmax(q_logits)
        for k in gaps:
            kl = kld_topk(p_lp[:, :k], np.take_along_axis(q_lp, idx[:, :k], axis=1)).kl[0]
            assert kl <= exact + 1e-12
            gaps[k].append(exact - kl)
    means = {k: float(np.mean(v)) for k, v in gaps.items()}
    assert means[8] > means[32] > means[128] >= 0


def test_t5_zero_probability_is_clamped_not_a_crash():
    p_lp = np.log(np.array([[0.7, 0.2, 0.05]]))
    q_lp = np.array([[-np.inf, np.log(0.5), np.log(0.4)]])
    res = kld_topk(p_lp, q_lp)
    assert np.isfinite(res.kl).all()
    assert res.clamps == 1


def test_t6_bootstrap_is_stable_and_contains_the_mean():
    rng = np.random.default_rng(3)
    counts = rng.integers(200, 500, size=60)
    sums = counts * rng.gamma(2.0, 0.01, size=60)
    a = bootstrap_mean_ci(sums, counts)
    b = bootstrap_mean_ci(sums, counts)
    assert a == b
    mean = sums.sum() / counts.sum()
    assert a[0] <= mean <= a[1]


def test_missing_ids_move_to_tail_and_stay_a_lower_bound():
    rng = np.random.default_rng(4)
    p_logits = rng.normal(size=(50, 256)) * 2
    q_logits = p_logits + rng.normal(size=(50, 256)) * 0.4
    idx, p_lp, _ = _topk(p_logits, 16)
    q_lp = np.take_along_axis(log_softmax(q_logits), idx, axis=1)
    full = kld_topk(p_lp, q_lp).kl
    q_missing = q_lp.copy()
    q_missing[:, 8:] = np.nan  # server did not report ranks 9-16
    partial = kld_topk(p_lp, q_missing).kl
    assert (partial <= full + 1e-12).all()
    assert (partial >= 0).all()


def test_ppl_ratio_one_for_same():
    x = np.log(np.array([0.2, 0.5, 0.1]))
    assert abs(ppl_ratio(x, x) - 1.0) < 1e-12
    assert ppl_ratio(x - 0.1, x) > 1.0
