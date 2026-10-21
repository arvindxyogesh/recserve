"""Statistical tests vendored from github.com/arvindxyogesh/ab-testing-framework.

Source commit 5f9175f ("final README polish"); files src/power/analysis.py,
src/frequentist/test.py and src/bayesian/test.py. Vendored rather than
installed because that repo is not packaged. Function bodies are unchanged
apart from formatting. Two behaviours to keep in mind when reading results:

  * two_proportion_z_test always reports a 95% CI on the difference,
    whatever ``alpha`` is.
  * beta_binomial_test's ``ci_lower`` / ``ci_upper`` are the 2.5th and 97.5th
    percentiles of the posterior difference (an equal-tailed credible
    interval), although the original docstring calls them an HDI.
  * two_proportion_z_test computes p as 1 - cdf(|z|), which is exactly 0
    for |z| above about 8.3; ab_stats.analyse also reports a precise value.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SampleSizeResult:
    n_per_group: int
    n_total: int
    mde: float  # minimum detectable effect
    power: float
    alpha: float


def sample_size_proportion(
    baseline_rate: float,
    mde: float,  # absolute lift (e.g. 0.02 for +2pp)
    power: float = 0.80,
    alpha: float = 0.05,
) -> SampleSizeResult:
    """Calculate required sample size per group for a two-proportion z-test."""
    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_bar = (p1 + p2) / 2

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = (
        (
            z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
            + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
        )
        / (p2 - p1)
    ) ** 2

    n = int(np.ceil(n))
    return SampleSizeResult(n_per_group=n, n_total=2 * n, mde=mde, power=power, alpha=alpha)


def power_curve(
    baseline_rate: float,
    n_per_group: int,
    mde_range: np.ndarray | None = None,
    alpha: float = 0.05,
) -> tuple:
    """Compute statistical power for a range of MDEs given a fixed sample size."""
    if mde_range is None:
        mde_range = np.linspace(0.001, 0.10, 50)

    powers = []
    for mde in mde_range:
        p1, p2 = baseline_rate, baseline_rate + mde
        p_bar = (p1 + p2) / 2
        se = np.sqrt(2 * p_bar * (1 - p_bar) / n_per_group)
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        non_central = abs(p2 - p1) / se
        power = 1 - stats.norm.cdf(z_alpha - non_central) + stats.norm.cdf(-z_alpha - non_central)
        powers.append(power)

    return mde_range, np.array(powers)


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    ci_lower: float
    ci_upper: float
    alpha: float

    def __str__(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.test_name}: stat={self.statistic:.4f}  "
            f"p={self.p_value:.4f}  [{sig}]  "
            f"effect={self.effect_size:.4f}  "
            f"95% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


def two_proportion_z_test(
    n_a: int,
    conv_a: int,
    n_b: int,
    conv_b: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> TestResult:
    """Z-test for the difference in two proportions.

    Args:
        n_a, conv_a: Sample size and conversions for group A.
        n_b, conv_b: Sample size and conversions for group B.
        alpha:       Significance level.
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        TestResult with z-statistic, p-value, 95% CI on Δp.
    """
    p_a = conv_a / n_a
    p_b = conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)

    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / (se_pool + 1e-12)

    if alternative == "two-sided":
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    elif alternative == "greater":
        p_value = 1 - stats.norm.cdf(z)
    else:
        p_value = stats.norm.cdf(z)

    # 95% CI on difference
    se_diff = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z95 = stats.norm.ppf(1 - 0.025)
    diff = p_b - p_a
    ci_low = diff - z95 * se_diff
    ci_high = diff + z95 * se_diff

    return TestResult(
        test_name="Two-Proportion Z-Test",
        statistic=z,
        p_value=p_value,
        significant=p_value < alpha,
        effect_size=diff,
        ci_lower=ci_low,
        ci_upper=ci_high,
        alpha=alpha,
    )


@dataclass
class BayesianResult:
    prob_b_better: float  # P(B > A | data)
    expected_lift: float  # E[(p_B - p_A)/p_A]
    ci_lower: float  # 95% HDI lower
    ci_upper: float  # 95% HDI upper
    p_a_posterior: np.ndarray  # posterior samples for A
    p_b_posterior: np.ndarray  # posterior samples for B

    def __str__(self) -> str:
        return (
            f"P(B>A)={self.prob_b_better:.4f}  "
            f"expected_lift={self.expected_lift:+.2%}  "
            f"95% HDI: [{self.ci_lower:+.4f}, {self.ci_upper:+.4f}]"
        )


def beta_binomial_test(
    n_a: int,
    conv_a: int,
    n_b: int,
    conv_b: int,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    n_samples: int = 100_000,
    seed: int = 42,
) -> BayesianResult:
    """Beta-Binomial Bayesian A/B test.

    Args:
        n_a, conv_a: Observations for variant A.
        n_b, conv_b: Observations for variant B.
        prior_alpha: Beta prior alpha (default: uniform).
        prior_beta:  Beta prior beta (default: uniform).
        n_samples:   MC samples for posterior.
        seed:        Random seed.

    Returns:
        BayesianResult with probability, lift, and credible interval.
    """
    rng = np.random.default_rng(seed)

    alpha_a = prior_alpha + conv_a
    beta_a = prior_beta + n_a - conv_a
    alpha_b = prior_alpha + conv_b
    beta_b = prior_beta + n_b - conv_b

    p_a = rng.beta(alpha_a, beta_a, size=n_samples)
    p_b = rng.beta(alpha_b, beta_b, size=n_samples)

    prob_b_better = float((p_b > p_a).mean())
    lift = (p_b - p_a) / (p_a + 1e-12)
    diff = p_b - p_a

    # 95% HDI (highest density interval) on difference
    diff_sorted = np.sort(diff)
    ci_low = float(np.percentile(diff_sorted, 2.5))
    ci_high = float(np.percentile(diff_sorted, 97.5))

    return BayesianResult(
        prob_b_better=prob_b_better,
        expected_lift=float(lift.mean()),
        ci_lower=ci_low,
        ci_upper=ci_high,
        p_a_posterior=p_a,
        p_b_posterior=p_b,
    )
