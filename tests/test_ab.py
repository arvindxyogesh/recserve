import numpy as np
import pytest

from recserve.experiments.ab_stats import analyse, assign
from recserve.experiments.abtest_vendored import power_curve, sample_size_proportion


def test_assignment_is_deterministic_and_balanced():
    ids = np.arange(20_000)
    a = assign(ids, "salt")
    np.testing.assert_array_equal(a, assign(ids, "salt"))
    assert abs(a.mean() - 0.5) < 0.02
    assert (a != assign(ids, "other-salt")).mean() > 0.4  # salt changes the split


def test_sample_size_matches_textbook_value():
    # p1 = 10%, p2 = 11%, alpha 0.05 two-sided, power 0.8 -> 14,751 per group
    assert sample_size_proportion(0.10, 0.01).n_per_group == 14_751


def test_power_curve_at_planned_n_is_target_power():
    plan = sample_size_proportion(0.2, 0.02)
    _, power = power_curve(0.2, plan.n_per_group, np.array([0.02]))
    assert power[0] == pytest.approx(0.8, abs=0.01)


def test_analyse_detects_large_lift_and_ci_brackets_truth():
    rng = np.random.default_rng(0)
    control = rng.random(20_000) < 0.20
    treatment = rng.random(20_000) < 0.26
    r = analyse(control, treatment, alpha=0.05, seed=0)
    assert r["significant"] and r["p_value"] < 1e-6
    assert r["rel_lift_ci"][0] < 0.30 < r["rel_lift_ci"][1]
    assert r["abs_diff_ci95"][0] < 0.06 < r["abs_diff_ci95"][1]
    assert r["bayes_prob_treatment_better"] > 0.999
    lo, hi = r["bayes_rel_lift_interval95"]
    assert lo < r["rel_lift"] < hi


def test_analyse_no_effect_is_not_significant():
    rng = np.random.default_rng(1)
    x = rng.random(40_000) < 0.3
    r = analyse(x[:20_000], x[20_000:], alpha=0.05, seed=0)
    assert not r["significant"]
    assert r["rel_lift_ci"][0] < 0 < r["rel_lift_ci"][1]
