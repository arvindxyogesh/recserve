"""Assignment and analysis for the simulated A/B test (pure numpy/scipy)."""

from __future__ import annotations

import hashlib

import numpy as np
from scipy import stats

from recserve.experiments.abtest_vendored import beta_binomial_test, two_proportion_z_test


def assign(user_ids: np.ndarray, salt: str) -> np.ndarray:
    """Deterministic 50/50 assignment: 0 = control, 1 = treatment."""
    return np.array(
        [int(hashlib.sha256(f"{salt}:{u}".encode()).hexdigest(), 16) % 2 for u in user_ids],
        dtype=np.int8,
    )


def analyse(control: np.ndarray, treatment: np.ndarray, alpha: float, seed: int) -> dict:
    n_c, n_t = len(control), len(treatment)
    x_c, x_t = int(control.sum()), int(treatment.sum())
    z = two_proportion_z_test(n_c, x_c, n_t, x_t, alpha=alpha)
    bayes = beta_binomial_test(n_c, x_c, n_t, x_t, seed=seed)
    p_c, p_t = x_c / n_c, x_t / n_t
    # Relative lift CI on the log risk ratio (delta method).
    se_log = np.sqrt((1 - p_t) / (n_t * p_t) + (1 - p_c) / (n_c * p_c))
    zq = stats.norm.ppf(1 - alpha / 2)
    log_rr = np.log(p_t / p_c)
    lift_post = (bayes.p_b_posterior - bayes.p_a_posterior) / bayes.p_a_posterior
    return {
        "control": {"n": n_c, "clicks": x_c, "rate": p_c},
        "treatment": {"n": n_t, "clicks": x_t, "rate": p_t},
        "abs_diff": z.effect_size,
        "abs_diff_ci95": [z.ci_lower, z.ci_upper],
        "z": z.statistic,
        "p_value": z.p_value,
        # The vendored test computes 1 - cdf, which underflows to 0 for large z.
        "p_value_precise": float(2 * stats.norm.sf(abs(z.statistic))),
        "significant": bool(z.significant),
        "rel_lift": p_t / p_c - 1,
        "rel_lift_ci": [
            float(np.exp(log_rr - zq * se_log) - 1),
            float(np.exp(log_rr + zq * se_log) - 1),
        ],
        "bayes_prob_treatment_better": bayes.prob_b_better,
        "bayes_expected_rel_lift": bayes.expected_lift,
        "bayes_abs_diff_interval95": [bayes.ci_lower, bayes.ci_upper],
        "bayes_rel_lift_interval95": [
            float(np.percentile(lift_post, 2.5)),
            float(np.percentile(lift_post, 97.5)),
        ],
    }
