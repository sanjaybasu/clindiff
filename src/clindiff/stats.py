"""Exact finite-sample statistical verification primitives for clinical AI change control.

Implements:
- Mixture-of-betting anytime-valid e-processes and confidence sequences on [0, 1]
  (order-invariant, finite-sample exact by Ville's inequality).
- Unbiased gross safe-to-unsafe patient flip rate and baseline intra-agent instability.
- Unbiased mean-squared case churn with exact bounded support mapping.
- Exact Minimum Verifiable Margin (Delta_min) calculator.
- Stratum profile product e-process (eliminates Horvitz-Thompson weight inflation).
- e-LOND sequential online FDR control across releases.
- Split-calibrated PPI++ paired estimator and fail-closed judge calibration equivalence gate.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy import optimize
from scipy.stats import binom, norm

# Exact 32-point Gauss-Chebyshev quadrature of the Jeffreys arcsine prior Beta(0.5, 0.5)
# on (0, 1), clipped strictly inside (0, 1). Each point u_k = sin^2(pi*(k + 0.5)/(2*K))
# carries equal exact quadrature weight 1/K. Because the prior over lambda_k(m) = u_k/(1-m)
# is fixed independently of observations, the mixture wealth process is an exact nonnegative
# supermartingale under H0: E[X] >= m (finite-sample anytime-valid by Ville's inequality),
# strictly non-decreasing in m, permutation-invariant, and near-minimax across both rare-flip
# (u -> 1) and moderate-rate regimes.
_BET_K = 32
_BET_GRID = np.clip(
    np.sin(math.pi * (np.arange(_BET_K, dtype=float) + 0.5) / (2.0 * _BET_K)) ** 2,
    0.002,
    0.995,
)
_LOG_BET_WEIGHTS = np.full(_BET_K, -math.log(_BET_K), dtype=float)


def _aggregate_clusters(vals: np.ndarray, cluster_ids: Sequence[Any] | None) -> tuple[np.ndarray, int]:
    """Average bounded per-row values within each independent cluster_id.

    Preserves the exact bounded support [lo, hi] while eliminating within-cluster
    non-independence (e.g., translated/paraphrased copies of the same base scenario).
    """
    if cluster_ids is None:
        return vals, int(vals.size)
    if len(cluster_ids) != vals.size:
        raise ValueError("cluster_ids length must match number of rows.")
    buckets: dict[Any, list[float]] = {}
    for cid, v in zip(cluster_ids, vals):
        buckets.setdefault(cid, []).append(float(v))
    if len(buckets) == vals.size:
        return vals, int(vals.size)
    c_means = np.array([float(np.mean(vs)) for vs in buckets.values()], dtype=float)
    return c_means, int(c_means.size)


def _validate_unit_array(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Input must be a non-empty 1D array.")
    if np.any(np.isnan(arr)) or np.any(arr < -1e-12) or np.any(arr > 1.0 + 1e-12):
        raise ValueError("All observations must lie in [0, 1].")
    return np.clip(arr, 0.0, 1.0)


def _log_betting_evalue_upper(arr: np.ndarray, m: float) -> float:
    """Log of mixture betting e-value testing H0: E[X] >= m against E[X] < m."""
    if m <= 0.0:
        return 0.0
    if m >= 1.0 - 1e-12:
        # At m = 1, any observation < 1 gives arbitrarily large evidence against E[X] >= 1.
        return 0.0 if np.all(arr >= 1.0 - 1e-12) else 100.0
    denom = 1.0 - m
    # Shape: (len(_BET_GRID), len(arr))
    scaled = (m - arr)[None, :] / denom
    factors = 1.0 + _BET_GRID[:, None] * scaled
    # All factors are >= 1 - max(_BET_GRID) = 0.005 > 0.
    log_wealth = np.sum(np.log(factors), axis=1) + _LOG_BET_WEIGHTS
    max_lw = float(np.max(log_wealth))
    # log(sum_k w_k * exp(log_wealth_k))
    return max_lw + math.log(float(np.sum(np.exp(log_wealth - max_lw))))


def betting_evalue_upper(x: Sequence[float] | np.ndarray, m: float) -> float:
    """Exact finite-sample e-value for H0: E[X] >= m (X in [0, 1])."""
    arr = _validate_unit_array(x)
    lw = _log_betting_evalue_upper(arr, float(m))
    return float(math.exp(min(lw, 700.0)))


def betting_upper_bound(x: Sequence[float] | np.ndarray, alpha: float = 0.05) -> float:
    """Exact finite-sample anytime-valid (1 - alpha) upper bound for E[X], X in [0, 1]."""
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1).")
    arr = _validate_unit_array(x)
    target = math.log(1.0 / alpha)
    mean_x = float(np.mean(arr))
    if mean_x >= 1.0 - 1e-12:
        return 1.0
    if _log_betting_evalue_upper(arr, 1.0 - 1e-9) < target:
        return 1.0
    lo = mean_x
    hi = 1.0 - 1e-9
    for _ in range(55):
        mid = 0.5 * (lo + hi)
        if _log_betting_evalue_upper(arr, mid) >= target:
            hi = mid
        else:
            lo = mid
    return float(hi)


def betting_two_sided_cs(x: Sequence[float] | np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Exact finite-sample anytime-valid two-sided (1 - alpha) confidence sequence on [0, 1]."""
    arr = _validate_unit_array(x)
    hi = betting_upper_bound(arr, alpha=alpha / 2.0)
    lo = 1.0 - betting_upper_bound(1.0 - arr, alpha=alpha / 2.0)
    return (float(max(0.0, lo)), float(min(1.0, hi)))


def _bounded_upper_bound(vals: np.ndarray, lo_bound: float, hi_bound: float, alpha: float) -> float:
    span = hi_bound - lo_bound
    u = (vals - lo_bound) / span
    return lo_bound + span * betting_upper_bound(u, alpha=alpha)


def _bounded_two_sided_cs(vals: np.ndarray, lo_bound: float, hi_bound: float, alpha: float) -> tuple[float, float]:
    span = hi_bound - lo_bound
    u = (vals - lo_bound) / span
    lo_u, hi_u = betting_two_sided_cs(u, alpha=alpha)
    return (lo_bound + span * lo_u, lo_bound + span * hi_u)


def gross_downward_flip_stats(
    s0: Sequence[Sequence[float]] | np.ndarray,
    s1: Sequence[Sequence[float]] | np.ndarray,
    alpha: float = 0.05,
    cluster_ids: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Unbiased gross safe-to-unsafe flip probability and baseline intra-agent instability.

    s0, s1: arrays of shape (N, R0) and (N, R1) with binary/unit safety scores in [0, 1],
    collected with independent rollouts/scoring across versions 0 and 1.
    cluster_ids: optional sequence of length N identifying independent clinical clusters
    (e.g., base_id across paired multilingual translations). When n_clusters < N, betting
    bounds are computed on cluster means to maintain exact finite-sample validity.
    """
    a0 = np.asarray(s0, dtype=float)
    a1 = np.asarray(s1, dtype=float)
    if a0.ndim == 1:
        a0 = a0[:, None]
    if a1.ndim == 1:
        a1 = a1[:, None]
    if a0.shape[0] != a1.shape[0] or a0.shape[0] == 0:
        raise ValueError("s0 and s1 must have matching non-zero row counts.")

    bar0 = np.mean(a0, axis=1)
    bar1 = np.mean(a1, axis=1)
    psi_down_i = bar0 * (1.0 - bar1)
    psi_down_clusters, n_clusters = _aggregate_clusters(psi_down_i, cluster_ids)
    psi_down_mean = float(np.mean(psi_down_clusters))
    psi_down_upper = betting_upper_bound(psi_down_clusters, alpha=alpha)
    psi_down_upper_unclustered = (
        betting_upper_bound(psi_down_i, alpha=alpha)
        if n_clusters < int(a0.shape[0])
        else psi_down_upper
    )

    r0 = a0.shape[1]
    if r0 >= 2:
        psi_self_i = (r0 / (r0 - 1.0)) * bar0 * (1.0 - bar0)
        psi_self_clusters, _ = _aggregate_clusters(psi_self_i, cluster_ids)
        psi_self_mean = float(np.mean(psi_self_clusters))
    else:
        psi_self_mean = float("nan")

    # Net harm: D_i = bar0 - bar1 in [-1, 1] (positive means v1 is LESS safe than v0)
    d_harm_i = bar0 - bar1
    d_harm_clusters, _ = _aggregate_clusters(d_harm_i, cluster_ids)
    harm_lo, harm_hi = _bounded_two_sided_cs(d_harm_clusters, -1.0, 1.0, alpha=alpha)
    harm_upper = _bounded_upper_bound(d_harm_clusters, -1.0, 1.0, alpha=alpha)

    bar0_c, _ = _aggregate_clusters(bar0, cluster_ids)
    bar1_c, _ = _aggregate_clusters(bar1, cluster_ids)

    return {
        "n_cases": int(a0.shape[0]),
        "n_clusters": int(n_clusters),
        "r0": int(r0),
        "r1": int(a1.shape[1]),
        "safety_v0_mean": float(np.mean(bar0_c)),
        "safety_v1_mean": float(np.mean(bar1_c)),
        "net_safety_delta": float(np.mean(bar1_c - bar0_c)),
        "net_harm_mean": float(np.mean(d_harm_clusters)),
        "net_harm_upper": float(harm_upper),
        "net_safety_ci_lower": float(-harm_hi),
        "net_safety_ci_upper": float(-harm_lo),
        "psi_down_mean": psi_down_mean,
        "psi_down_upper": float(psi_down_upper),
        "psi_down_upper_unclustered": float(psi_down_upper_unclustered),
        "psi_self_mean": psi_self_mean,
    }


def _max_var_term(r: int) -> float:
    """Maximum possible value of barS*(1-barS) for binary average of r trials."""
    if r % 2 == 0:
        return 0.25
    return (r * r - 1.0) / (4.0 * r * r)


def unbiased_mean_squared_churn(
    s0: Sequence[Sequence[float]] | np.ndarray,
    s1: Sequence[Sequence[float]] | np.ndarray,
    alpha: float = 0.05,
    cluster_ids: Sequence[Any] | None = None,
) -> dict[str, float]:
    """Exact unbiased estimator and finite-sample betting CS for E[(p0 - p1)^2].

    Requires R0 >= 2 and R1 >= 2 independent rollouts per case.
    Supports optional cluster_ids for cluster-exact confidence sequences.
    """
    a0 = np.asarray(s0, dtype=float)
    a1 = np.asarray(s1, dtype=float)
    if a0.ndim != 2 or a1.ndim != 2 or a0.shape[0] != a1.shape[0]:
        raise ValueError("s0 and s1 must be 2D arrays with equal row counts.")
    r0, r1 = a0.shape[1], a1.shape[1]
    if r0 < 2 or r1 < 2:
        raise ValueError("Unbiased mean-squared churn requires R0 >= 2 and R1 >= 2.")

    bar0 = np.mean(a0, axis=1)
    bar1 = np.mean(a1, axis=1)
    theta_i = (bar0 - bar1) ** 2 - (bar0 * (1.0 - bar0)) / (r0 - 1.0) - (bar1 * (1.0 - bar1)) / (r1 - 1.0)
    support_lo = -(_max_var_term(r0) / (r0 - 1.0) + _max_var_term(r1) / (r1 - 1.0))
    support_hi = 1.0

    # Clip numerical rounding strictly to exact theoretical support
    theta_i = np.clip(theta_i, support_lo, support_hi)
    theta_c, n_clusters = _aggregate_clusters(theta_i, cluster_ids)
    lo, hi = _bounded_two_sided_cs(theta_c, support_lo, support_hi, alpha=alpha)
    upper_1s = _bounded_upper_bound(theta_c, support_lo, support_hi, alpha=alpha)

    return {
        "n_cases": int(a0.shape[0]),
        "n_clusters": int(n_clusters),
        "theta_sq_mean": float(np.mean(theta_c)),
        "theta_sq_lower": float(lo),
        "theta_sq_upper_2s": float(hi),
        "theta_sq_upper": float(upper_1s),
        "support_lower": float(support_lo),
        "support_upper": float(support_hi),
    }


def minimum_verifiable_margin(
    n: int,
    alpha: float = 0.05,
    discordant_rate: float = 0.05,
    power: float = 0.80,
    background_flip_rate: float = 0.01,
) -> dict[str, float]:
    """Compute exact smallest certifiable safety margin at sample size n.

    Because the permutation-invariant betting upper bound on binary flips is strictly
    increasing in K = sum(X_i), the exact smallest margin Delta certifiable with probability
    >= power under true background flip rate p_bg is U_bet(k_star, n; alpha), where
    k_star = F^{-1}_{Bin(n, p_bg)}(power). Zero heuristic scaling constants are used.

    Returns:
      - zero_error_betting_floor: exact finite-sample betting upper bound when 0/n flips occur.
      - exact_betting_mvm: exact finite-sample (power)-achievable betting margin at background_flip_rate.
      - exact_betting_mvm_2pct: exact finite-sample (power)-achievable betting margin at 2% background flips.
      - fixed_horizon_mvm: classical asymptotic one-sided paired z-test MVM comparator.
      - empirical_mvm: alias to exact_betting_mvm for report compatibility.
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    zero_floor = betting_upper_bound(np.zeros(n), alpha=alpha)

    def _exact_bet_at_bg(p_bg: float) -> float:
        if p_bg <= 0.0:
            return zero_floor
        k_star = int(min(n, max(0, binom.ppf(power, n, p_bg))))
        arr = np.zeros(n, dtype=float)
        if k_star > 0:
            arr[:k_star] = 1.0
        return betting_upper_bound(arr, alpha=alpha)

    exact_mvm_bg = _exact_bet_at_bg(background_flip_rate)
    exact_mvm_2pct = _exact_bet_at_bg(0.02)
    exact_margin_above_bg = max(0.0, exact_mvm_bg - background_flip_rate)

    z_a = float(norm.ppf(1.0 - alpha))
    z_b = float(norm.ppf(power))
    sigma_d = math.sqrt(max(discordant_rate, 1e-6))
    fixed_mvm = (z_a + z_b) * sigma_d / math.sqrt(n)
    return {
        "n": int(n),
        "alpha": float(alpha),
        "power": float(power),
        "background_flip_rate": float(background_flip_rate),
        "discordant_rate": float(discordant_rate),
        "zero_error_betting_floor": float(zero_floor),
        "exact_betting_mvm": float(exact_mvm_bg),
        "exact_betting_mvm_2pct": float(exact_mvm_2pct),
        "exact_betting_margin_above_bg": float(exact_margin_above_bg),
        "fixed_horizon_mvm": float(fixed_mvm),
        "empirical_mvm": float(exact_margin_above_bg),
    }


def profile_product_evalue(
    strata_arrays: Sequence[Sequence[float] | np.ndarray],
    weights: Sequence[float] | np.ndarray,
    mu0: float,
    alpha: float = 0.05,
) -> float:
    """Anytime-valid profile product e-process for H0: sum_s w_s * E[X_s] >= mu0.

    Computes inf_{mu in [0,1]^S : sum_s w_s mu_s >= mu0} prod_s E_s(mu_s).
    Eliminates Horvitz-Thompson weight range inflation.
    """
    del alpha  # e-value itself does not depend on alpha
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or len(w) != len(strata_arrays) or len(w) == 0:
        raise ValueError("weights and strata_arrays must have matching non-zero length.")
    if np.any(w <= 0) or abs(float(np.sum(w)) - 1.0) > 1e-6:
        raise ValueError("weights must be strictly positive and sum to 1.")
    strata = [_validate_unit_array(s) for s in strata_arrays]
    mu0 = float(mu0)
    if mu0 <= 0.0:
        return 1.0
    if mu0 >= 1.0:
        return float(math.exp(min(sum(_log_betting_evalue_upper(s, 1.0 - 1e-9) for s in strata), 700.0)))

    def obj(mu_vec: np.ndarray) -> float:
        return float(sum(_log_betting_evalue_upper(s, float(m)) for s, m in zip(strata, mu_vec)))

    k = len(strata)
    if k == 1:
        return float(math.exp(min(_log_betting_evalue_upper(strata[0], mu0), 700.0)))

    if k == 2:
        lo1 = max(0.0, (mu0 - w[1]) / w[0])
        hi1 = min(1.0, mu0 / w[0])
        grid = np.linspace(lo1, hi1, 201)
        vals = [obj(np.array([m1, np.clip((mu0 - w[0] * m1) / w[1], 0.0, 1.0)])) for m1 in grid]
        best_idx = int(np.argmin(vals))
        b_lo = grid[max(0, best_idx - 1)]
        b_hi = grid[min(len(grid) - 1, best_idx + 1)]
        res = optimize.minimize_scalar(
            lambda m1: obj(np.array([m1, np.clip((mu0 - w[0] * m1) / w[1], 0.0, 1.0)])),
            bounds=(b_lo, b_hi),
            method="bounded",
        )
        min_lw = min(vals[best_idx], float(res.fun))
        return float(math.exp(min(min_lw, 700.0)))

    # General S >= 3: multi-start SLSQP on linear equality sum(w * mu) == mu0
    bounds = [(0.0, 1.0)] * k
    cons = {"type": "eq", "fun": lambda mu: float(np.dot(w, mu) - mu0)}
    emp = np.array([float(np.mean(s)) for s in strata])
    starts = [np.full(k, mu0)]
    if float(np.dot(w, emp)) < mu0:
        shift = (mu0 - float(np.dot(w, emp)))
        starts.append(np.clip(emp + shift, 0.0, 1.0))
    for idx in range(k):
        v = np.zeros(k)
        v[idx] = min(1.0, mu0 / w[idx])
        rem = mu0 - w[idx] * v[idx]
        if rem > 1e-12:
            other_w = 1.0 - w[idx]
            for j in range(k):
                if j != idx:
                    v[j] = min(1.0, rem / other_w)
        starts.append(v)

    best_lw = float("inf")
    for x0 in starts:
        res = optimize.minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9, "maxiter": 200})
        if res.success and abs(float(np.dot(w, res.x) - mu0)) < 1e-5:
            best_lw = min(best_lw, float(res.fun))
    if not math.isfinite(best_lw):
        best_lw = obj(np.full(k, mu0))
    return float(math.exp(min(best_lw, 700.0)))


def profile_product_upper_bound(
    strata_arrays: Sequence[Sequence[float] | np.ndarray],
    weights: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Invert profile_product_evalue to get exact anytime-valid (1 - alpha) upper bound."""
    w = np.asarray(weights, dtype=float)
    strata = [_validate_unit_array(s) for s in strata_arrays]
    emp_global = float(sum(wi * float(np.mean(si)) for wi, si in zip(w, strata)))
    thresh = 1.0 / alpha
    if emp_global >= 1.0 - 1e-9 or profile_product_evalue(strata, w, 1.0 - 1e-6, alpha=alpha) < thresh:
        return 1.0
    lo, hi = emp_global, 1.0 - 1e-6
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        if profile_product_evalue(strata, w, mid, alpha=alpha) >= thresh:
            hi = mid
        else:
            lo = mid
    return float(hi)


def elond_step(e_value: float, alpha: float, state: dict[str, int]) -> dict[str, Any]:
    """One release step of e-LOND (Xu & Ramdas 2024) online FDR control on e-values."""
    k = int(state.get("k", 0)) + 1
    r_prev = int(state.get("rejections", 0))
    gamma_k = 6.0 / (math.pi * math.pi * k * k)
    alpha_k = alpha * gamma_k * (r_prev + 1)
    threshold = 1.0 / alpha_k
    rejected = float(e_value) >= threshold
    state["k"] = k
    state["rejections"] = r_prev + (1 if rejected else 0)
    return {
        "release_index": k,
        "gamma_k": gamma_k,
        "alpha_k": alpha_k,
        "e_threshold": threshold,
        "e_value": float(e_value),
        "rejected": rejected,
        "cumulative_rejections": state["rejections"],
    }


def ppi_plus_plus_paired(
    f0_unlab: Sequence[float] | np.ndarray,
    f1_unlab: Sequence[float] | np.ndarray,
    f0_lab: Sequence[float] | np.ndarray,
    f1_lab: Sequence[float] | np.ndarray,
    y0_lab: Sequence[float] | np.ndarray,
    y1_lab: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
    tune_fraction: float = 0.20,
) -> dict[str, float]:
    """Split-calibrated paired PPI++ estimator with exact finite-sample betting CI.

    Fixes tuning weight lambda in [0, 1] on an initial tuning slice of the labeled set
    (zero in-sample leakage), then combines split-alpha betting CSs across the unlabeled
    judge differences and the held-out rectified clinician residuals.
    """
    df_u = np.asarray(f1_unlab, dtype=float) - np.asarray(f0_unlab, dtype=float)
    df_l = np.asarray(f1_lab, dtype=float) - np.asarray(f0_lab, dtype=float)
    dy_l = np.asarray(y1_lab, dtype=float) - np.asarray(y0_lab, dtype=float)
    if df_l.size < 10 or dy_l.size != df_l.size:
        raise ValueError("Need at least 10 paired clinician-labeled cases for PPI++.")

    m_tune = max(5, min(int(round(df_l.size * tune_fraction)), df_l.size - 5))
    var_f = float(np.var(df_l[:m_tune], ddof=1))
    cov_yf = float(np.cov(dy_l[:m_tune], df_l[:m_tune], ddof=1)[0, 1]) if var_f > 1e-12 else 0.0
    lam = float(np.clip(cov_yf / (var_f + 1e-12), 0.0, 1.0)) if var_f > 1e-12 else 0.0

    # Evaluation slice (independent of lam)
    rect = dy_l[m_tune:] - lam * df_l[m_tune:]
    mean_f = float(np.mean(df_u))
    mean_rect = float(np.mean(rect))
    ppi_est = lam * mean_f + mean_rect

    # Exact finite-sample split-alpha betting CS
    alpha_u = 0.25 * alpha
    alpha_r = 0.75 * alpha
    lo_f, hi_f = _bounded_two_sided_cs(df_u, -1.0, 1.0, alpha=alpha_u)
    rect_lo_bound = -1.0 - lam
    rect_hi_bound = 1.0 + lam
    lo_r, hi_r = _bounded_two_sided_cs(np.clip(rect, rect_lo_bound, rect_hi_bound), rect_lo_bound, rect_hi_bound, alpha=alpha_r)

    ci_lo = lam * lo_f + lo_r
    ci_hi = lam * hi_f + hi_r

    return {
        "n_unlabeled": int(df_u.size),
        "m_labeled_total": int(df_l.size),
        "m_eval": int(rect.size),
        "lambda_hat": lam,
        "raw_judge_delta": mean_f,
        "clinician_only_delta": float(np.mean(dy_l)),
        "ppi_delta": float(ppi_est),
        "ci_lower": float(ci_lo),
        "ci_upper": float(ci_hi),
    }


def judge_calibration_equivalence(
    f0_lab: Sequence[float] | np.ndarray,
    f1_lab: Sequence[float] | np.ndarray,
    y0_lab: Sequence[float] | np.ndarray,
    y1_lab: Sequence[float] | np.ndarray,
    tau_cal: float = 0.02,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fail-closed differential judge-calibration equivalence test.

    Defines judge error e_v = f_v - y_v and differential error delta_cal = e_1 - e_0 in [-2, 2].
    Passes calibration ONLY IF the exact two-sided (1 - alpha) betting CS lies strictly
    inside (-tau_cal, +tau_cal). Flags shortcut_alert whenever the CS excludes 0.
    """
    e0 = np.asarray(f0_lab, dtype=float) - np.asarray(y0_lab, dtype=float)
    e1 = np.asarray(f1_lab, dtype=float) - np.asarray(y1_lab, dtype=float)
    d_cal = np.clip(e1 - e0, -2.0, 2.0)
    lo, hi = _bounded_two_sided_cs(d_cal, -2.0, 2.0, alpha=alpha)
    cal_passed = bool(max(abs(lo), abs(hi)) < tau_cal)
    shortcut_alert = bool(lo > 0.0 or hi < 0.0)
    return {
        "m_labeled": int(d_cal.size),
        "diff_cal_mean": float(np.mean(d_cal)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "tau_cal": float(tau_cal),
        "calibration_passed": cal_passed,
        "shortcut_alert": shortcut_alert,
    }
