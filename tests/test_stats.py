"""Tests for clindiff.stats — written first under strict TDD."""
import numpy as np
import pytest

from clindiff.stats import (
    betting_upper_bound,
    betting_two_sided_cs,
    betting_evalue_upper,
    gross_downward_flip_stats,
    unbiased_mean_squared_churn,
    minimum_verifiable_margin,
    profile_product_evalue,
    profile_product_upper_bound,
    elond_step,
    ppi_plus_plus_paired,
    judge_calibration_equivalence,
)


def test_betting_upper_bound_zero_errors_matches_finite_sample_ceiling():
    """With zero errors in N=300 at alpha=0.05, upper bound is strictly above 0 and near ln(20)/300."""
    x = np.zeros(300)
    ub = betting_upper_bound(x, alpha=0.05)
    assert 0.010 <= ub <= 0.022
    assert ub > 0.005  # proves 0.5 pp cannot be certified at N=300


def test_betting_two_sided_cs_contains_true_mean_empirical_coverage():
    """Monte Carlo check: anytime-valid betting CS maintains >= 1-alpha coverage across seeds."""
    rng = np.random.default_rng(20261123)
    true_p = 0.08
    misses = 0
    trials = 250
    for _ in range(trials):
        x = rng.binomial(1, true_p, size=200).astype(float)
        lo, hi = betting_two_sided_cs(x, alpha=0.10)
        if not (lo <= true_p <= hi):
            misses += 1
    assert misses / trials <= 0.12  # conservative finite-sample guarantee


def test_betting_rejects_out_of_range_inputs():
    with pytest.raises(ValueError):
        betting_upper_bound(np.array([-0.1, 0.5]), alpha=0.05)
    with pytest.raises(ValueError):
        betting_upper_bound(np.array([0.2, 1.2]), alpha=0.05)


def test_gross_downward_flip_and_self_instability_unbiased():
    """Verify E[psi_down] = E[p0*(1-p1)] and E[psi_self] = E[p0*(1-p0)] under independent rollouts."""
    rng = np.random.default_rng(42)
    n, r = 8000, 3
    p0 = rng.uniform(0.3, 0.9, size=n)
    p1 = np.clip(p0 - 0.1, 0.0, 1.0)
    s0 = rng.binomial(1, p0[:, None], size=(n, r)).astype(float)
    s1 = rng.binomial(1, p1[:, None], size=(n, r)).astype(float)
    res = gross_downward_flip_stats(s0, s1, alpha=0.05)
    true_down = float(np.mean(p0 * (1.0 - p1)))
    true_self = float(np.mean(p0 * (1.0 - p0)))
    assert abs(res["psi_down_mean"] - true_down) < 0.015
    assert abs(res["psi_self_mean"] - true_self) < 0.015
    assert res["psi_down_upper"] >= true_down - 0.005


def test_unbiased_churn_catches_symmetric_churn_where_net_change_is_zero():
    """Half improve 0.5->1.0, half collapse 0.5->0.0: net delta is 0, true churn is 0.25."""
    rng = np.random.default_rng(7)
    n, r = 4000, 3
    p0 = np.full(n, 0.5)
    p1 = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    s0 = rng.binomial(1, p0[:, None], size=(n, r)).astype(float)
    s1 = rng.binomial(1, p1[:, None], size=(n, r)).astype(float)
    res = unbiased_mean_squared_churn(s0, s1, alpha=0.05)
    assert abs(res["theta_sq_mean"] - 0.25) < 0.02
    assert res["support_lower"] == pytest.approx(-2.0 / 9.0, abs=1e-12)
    assert res["theta_sq_lower"] > 0.15  # confidently detects hidden churn


def test_unbiased_churn_is_zero_when_versions_are_identical():
    """Identical stochastic versions (p0 == p1 == 0.5) must yield ~0 unbiased churn."""
    rng = np.random.default_rng(11)
    n, r = 6000, 3
    s0 = rng.binomial(1, 0.5, size=(n, r)).astype(float)
    s1 = rng.binomial(1, 0.5, size=(n, r)).astype(float)
    res = unbiased_mean_squared_churn(s0, s1, alpha=0.05)
    assert abs(res["theta_sq_mean"]) < 0.02


def test_unbiased_churn_requires_at_least_two_repeats():
    with pytest.raises(ValueError):
        unbiased_mean_squared_churn(np.ones((10, 1)), np.ones((10, 3)))


def test_minimum_verifiable_margin_shows_small_samples_cannot_certify_half_percent():
    """At N=30 or N=100, MVM is large; at N=949, 0.5 pp is still impossible at K=12."""
    m30 = minimum_verifiable_margin(n=30, alpha=0.05, discordant_rate=0.05)
    m100 = minimum_verifiable_margin(n=100, alpha=0.05, discordant_rate=0.05)
    m949_k12 = minimum_verifiable_margin(n=949, alpha=0.05 / 12, discordant_rate=0.05)
    m25000_k12 = minimum_verifiable_margin(n=25000, alpha=0.05 / 12, discordant_rate=0.05)
    assert m30["zero_error_betting_floor"] > 0.09
    assert m100["zero_error_betting_floor"] > 0.025
    assert m949_k12["empirical_mvm"] > 0.015
    assert m25000_k12["empirical_mvm"] < 0.007


def test_profile_product_evalue_valid_under_null_and_monotone():
    """Profile product e-process stays valid under H0, rejects H1, and is monotone in mu0."""
    rng = np.random.default_rng(99)
    # 2 strata: 10% high-risk (oversampled n=200), 90% routine (n=200)
    weights = np.array([0.10, 0.90])
    strata_alt = [
        rng.binomial(1, 0.04, size=200).astype(float),
        rng.binomial(1, 0.005, size=200).astype(float),
    ]
    e_lo = profile_product_evalue(strata_alt, weights, mu0=0.04, alpha=0.05)
    e_hi = profile_product_evalue(strata_alt, weights, mu0=0.06, alpha=0.05)
    assert e_hi > e_lo
    assert e_hi > 20.0  # rejects H0: global error >= 6%

    ub = profile_product_upper_bound(strata_alt, weights, alpha=0.05)
    emp = 0.10 * float(np.mean(strata_alt[0])) + 0.90 * float(np.mean(strata_alt[1]))
    assert emp < ub < 0.06

    # Under H0: true global error = 0.05, test H0: mu >= 0.05 across 80 trials
    false_rejects = 0
    for _ in range(80):
        strata_null = [
            rng.binomial(1, 0.14, size=100).astype(float),
            rng.binomial(1, 0.04, size=100).astype(float),
        ]
        if profile_product_evalue(strata_null, weights, mu0=0.05, alpha=0.05) >= 20.0:
            false_rejects += 1
    assert false_rejects / 80 <= 0.08


def test_elond_controls_sequential_thresholds():
    """e-LOND alpha_k scales with (R_{k-1} + 1) * gamma_k and tracks rejections."""
    state = {"k": 0, "rejections": 0}
    # Release 1: weak e-value -> not rejected
    r1 = elond_step(e_value=5.0, alpha=0.05, state=state)
    assert r1["rejected"] is False
    assert state["k"] == 1 and state["rejections"] == 0
    # Release 2: strong e-value -> rejected
    r2 = elond_step(e_value=200.0, alpha=0.05, state=state)
    assert r2["rejected"] is True
    assert state["k"] == 2 and state["rejections"] == 1


def test_ppi_plus_plus_recovers_true_delta_and_catches_judge_shortcut():
    """When judge is inflated by +6 pp on v1 (shortcut), PPI++ stays unbiased and calibration fails closed."""
    rng = np.random.default_rng(123)
    n_unlab, m_lab = 1000, 150
    # True quality delta = 0.00 (no true improvement)
    y0_all = rng.binomial(1, 0.70, size=n_unlab + m_lab).astype(float)
    y1_all = y0_all.copy()  # identical true quality
    # Judge agrees closely on v0, but has a verbosity shortcut inflating v1 false positives by ~10 pp
    f0_all = np.where(rng.random(n_unlab + m_lab) < 0.90, y0_all, 1.0 - y0_all)
    f1_all = f0_all.copy()
    flip_idx = (y1_all == 0) & (f1_all == 0) & (rng.random(n_unlab + m_lab) < 0.45)
    f1_all[flip_idx] = 1.0

    res_ppi = ppi_plus_plus_paired(
        f0_unlab=f0_all[:n_unlab],
        f1_unlab=f1_all[:n_unlab],
        f0_lab=f0_all[n_unlab:],
        f1_lab=f1_all[n_unlab:],
        y0_lab=y0_all[n_unlab:],
        y1_lab=y1_all[n_unlab:],
        alpha=0.05,
    )
    # Raw judge looks inflated (> +0.05), but PPI++ stays near 0 and CI covers 0
    assert res_ppi["raw_judge_delta"] > 0.04
    assert abs(res_ppi["ppi_delta"]) < 0.04
    assert res_ppi["ci_lower"] <= 0.0 <= res_ppi["ci_upper"]

    cal = judge_calibration_equivalence(
        f0_lab=f0_all[n_unlab:],
        f1_lab=f1_all[n_unlab:],
        y0_lab=y0_all[n_unlab:],
        y1_lab=y1_all[n_unlab:],
        tau_cal=0.02,
        alpha=0.05,
    )
    assert cal["calibration_passed"] is False
    assert cal["shortcut_alert"] is True


def test_cluster_exact_bounds_widen_properly_under_replicated_clusters():
    """Replicating 80 independent clusters 3x (N=240) must give identical cluster-exact UB to N=80."""
    rng = np.random.default_rng(77)
    s0_80 = np.ones((80, 3), dtype=float)
    s1_80 = np.ones((80, 3), dtype=float)
    s1_80[0] = [0, 0, 0]
    s1_80[1] = [0, 1, 1]
    res_80 = gross_downward_flip_stats(s0_80, s1_80, alpha=0.05)

    # Duplicate each scenario 3x with matching cluster_id (like en_std / en_lowlit / es_spanish)
    s0_240 = np.repeat(s0_80, 3, axis=0)
    s1_240 = np.repeat(s1_80, 3, axis=0)
    cids = [f"base_{i}" for i in range(80) for _ in range(3)]
    res_240_c = gross_downward_flip_stats(s0_240, s1_240, alpha=0.05, cluster_ids=cids)

    assert res_240_c["n_cases"] == 240
    assert res_240_c["n_clusters"] == 80
    assert abs(res_240_c["psi_down_upper"] - res_80["psi_down_upper"]) < 1e-12
    assert res_240_c["psi_down_upper"] > res_240_c["psi_down_upper_unclustered"]


def test_exact_betting_mvm_uses_zero_heuristic_and_matches_binomial_quantile():
    """Exact betting MVM equals betting_upper_bound at k_star = binom.ppf(power, n, p_bg)."""
    from scipy.stats import binom
    for n in (50, 80, 300, 949):
        m = minimum_verifiable_margin(n=n, alpha=0.05, background_flip_rate=0.01, power=0.80)
        k_star = int(binom.ppf(0.80, n, 0.01))
        expected = betting_upper_bound(np.concatenate([np.ones(k_star), np.zeros(n - k_star)]), alpha=0.05)
        assert abs(m["exact_betting_mvm"] - expected) < 1e-12
        assert abs(m["exact_betting_margin_above_bg"] - max(0.0, expected - 0.01)) < 1e-12


def test_betting_anytime_valid_type1_error_monte_carlo():
    """Monte Carlo verification: sequential anytime-valid Type I error stays <= alpha=0.05."""
    rng = np.random.default_rng(20261124)
    for true_p in (0.025, 0.05, 0.15):
        n_paths, horizon = 600, 150
        x = rng.binomial(1, true_p, size=(n_paths, horizon)).astype(float)
        # Check anytime rejection at stopping checkpoints t = 25, 50, 100, 150
        rej = 0
        for i in range(n_paths):
            hit = False
            for t in (25, 50, 100, 150):
                if betting_evalue_upper(x[i, :t], m=true_p) >= 20.0:
                    hit = True
                    break
            if hit:
                rej += 1
        assert rej / n_paths <= 0.055  # finite-sample exact <= 0.05 (+ tiny MC slack)

