"""Release gate orchestration (Gates 0, 1, 2, 3) and cryptographic manifest signing."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .stats import (
    betting_evalue_upper,
    betting_upper_bound,
    elond_step,
    gross_downward_flip_stats,
    unbiased_mean_squared_churn,
    minimum_verifiable_margin,
    ppi_plus_plus_paired,
    judge_calibration_equivalence,
    profile_product_upper_bound,
)

PROFILES_DIR = Path(__file__).resolve().parents[2] / "profiles"


def hash_manifest(obj: dict[str, Any]) -> str:
    """Canonical SHA-256 hash of a JSON-serializable configuration or report dict."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_profile(profile_name: str) -> dict[str, Any]:
    path = PROFILES_DIR / f"{profile_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Unknown governance profile: {profile_name} ({path})")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _extract_cluster_ids(records: Sequence[dict[str, Any]]) -> list[Any] | None:
    if any("cluster_id" in r for r in records):
        return [r.get("cluster_id", r.get("case_id", i)) for i, r in enumerate(records)]
    if any("base_id" in r for r in records):
        return [r.get("base_id", r.get("case_id", i)) for i, r in enumerate(records)]
    return None


def _cluster_psi_down_vector(records: Sequence[dict[str, Any]]) -> np.ndarray:
    s0 = np.array([r["safe_v0"] for r in records], dtype=float)
    s1 = np.array([r["safe_v1"] for r in records], dtype=float)
    if s0.ndim == 1:
        s0 = s0[:, None]
    if s1.ndim == 1:
        s1 = s1[:, None]
    psi_i = np.mean(s0, axis=1) * (1.0 - np.mean(s1, axis=1))
    cids = _extract_cluster_ids(records)
    if cids is None:
        return psi_i
    buckets: dict[Any, list[float]] = {}
    for c, v in zip(cids, psi_i):
        buckets.setdefault(c, []).append(float(v))
    return np.array([float(np.mean(vs)) for vs in buckets.values()], dtype=float)


def evaluate_release(
    records: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
    profile_name: str = "enterprise_governance",
    clinician_calibration_records: Sequence[dict[str, Any]] | None = None,
    threshold_overrides: dict[str, float] | None = None,
    domain_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run Gates 0-3 on paired multi-run trace records and return a signed release report."""
    if not records:
        raise ValueError("records must be non-empty.")

    prof = load_profile(profile_name)
    th = dict(prof["thresholds"])
    if threshold_overrides:
        th.update(threshold_overrides)

    alpha = float(th["alpha_safety"])
    all_cids = _extract_cluster_ids(records)

    # Gate 0: Deterministic hard safety / scope invariants
    g0_fail_v0 = sum(1 for r in records if not bool(r.get("gate0_v0_pass", True)))
    g0_fail_v1 = sum(1 for r in records if not bool(r.get("gate0_v1_pass", True)))
    g0_passed = g0_fail_v1 <= int(th["gate0_max_critical_failures"])
    gate0 = {
        "n_cases": len(records),
        "failures_v0": g0_fail_v0,
        "failures_v1": g0_fail_v1,
        "max_allowed": int(th["gate0_max_critical_failures"]),
        "passed": g0_passed,
    }

    # Gate 1: Paired safety, downward flip, unbiased churn, anchor drift, and subgroups
    s0 = np.array([r["safe_v0"] for r in records], dtype=float)
    s1 = np.array([r["safe_v1"] for r in records], dtype=float)
    overall_flips = gross_downward_flip_stats(s0, s1, alpha=alpha, cluster_ids=all_cids)
    overall_churn = (
        unbiased_mean_squared_churn(s0, s1, alpha=alpha, cluster_ids=all_cids)
        if s0.ndim == 2 and s0.shape[1] >= 2 and s1.shape[1] >= 2
        else None
    )

    # Discordant rate on mean paired scores for empirical MVM (uses independent cluster count)
    disc_rate = float(np.mean(np.abs(np.mean(s1, axis=1) - np.mean(s0, axis=1)) > 1e-9)) if s0.ndim == 2 else float(np.mean(s1 != s0))
    mvm = minimum_verifiable_margin(n=int(overall_flips["n_clusters"]), alpha=alpha, discordant_rate=max(disc_rate, 0.02))

    # Cumulative anchor check against locked baseline A0 (if supplied)
    anchor_res = None
    if all("safe_anchor" in r for r in records):
        sa = np.array([r["safe_anchor"] for r in records], dtype=float)
        anchor_flips = gross_downward_flip_stats(sa, s1, alpha=alpha, cluster_ids=all_cids)
        anchor_res = {
            "net_harm_vs_anchor": anchor_flips["net_harm_mean"],
            "net_harm_upper_vs_anchor": anchor_flips["net_harm_upper"],
            "psi_down_vs_anchor_mean": anchor_flips["psi_down_mean"],
            "psi_down_vs_anchor_upper": anchor_flips["psi_down_upper"],
            "passed": anchor_flips["net_harm_mean"] <= float(th["anchor_max_drift"]),
        }

    # Subgroup breakdown by stratum (language, literacy, buyer canary)
    subgroups: dict[str, Any] = {}
    strata_names = sorted({str(r.get("stratum", "all")) for r in records})
    for st in strata_names:
        sub = [r for r in records if str(r.get("stratum", "all")) == st]
        ss0 = np.array([r["safe_v0"] for r in sub], dtype=float)
        ss1 = np.array([r["safe_v1"] for r in sub], dtype=float)
        sub_cids = _extract_cluster_ids(sub)
        sf = gross_downward_flip_stats(ss0, ss1, alpha=alpha, cluster_ids=sub_cids)
        sc = (
            unbiased_mean_squared_churn(ss0, ss1, alpha=alpha, cluster_ids=sub_cids)
            if ss0.ndim == 2 and ss0.shape[1] >= 2 and ss1.shape[1] >= 2
            else None
        )
        subgroups[st] = {
            "n_cases": len(sub),
            "n_clusters": sf["n_clusters"],
            "safety_v0_mean": sf["safety_v0_mean"],
            "safety_v1_mean": sf["safety_v1_mean"],
            "net_safety_delta": sf["net_safety_delta"],
            "psi_down_mean": sf["psi_down_mean"],
            "psi_down_upper": sf["psi_down_upper"],
            "theta_sq_mean": sc["theta_sq_mean"] if sc else None,
        }

    # Aggregate quality check (if distinct quality scores are provided)
    q0 = np.array([r.get("quality_v0", r["safe_v0"]) for r in records], dtype=float)
    q1 = np.array([r.get("quality_v1", r["safe_v1"]) for r in records], dtype=float)
    q_stats = gross_downward_flip_stats(q0, q1, alpha=alpha, cluster_ids=all_cids)

    # Optional safety-positive vs. routine stratum split (prevents routine alert-burden
    # increases from being conflated with missed clinical safety hazards)
    stratum_split: dict[str, Any] | None = None
    has_sp_flag = any("is_safety_positive" in r for r in records)
    sp_recs = [r for r in records if bool(r.get("is_safety_positive"))] if has_sp_flag else []
    rt_recs = [r for r in records if not bool(r.get("is_safety_positive"))] if has_sp_flag else []
    if sp_recs and rt_recs:
        sp_s0 = np.array([r["safe_v0"] for r in sp_recs], dtype=float)
        sp_s1 = np.array([r["safe_v1"] for r in sp_recs], dtype=float)
        rt_s0 = np.array([r["safe_v0"] for r in rt_recs], dtype=float)
        rt_s1 = np.array([r["safe_v1"] for r in rt_recs], dtype=float)
        sp_cids = _extract_cluster_ids(sp_recs)
        rt_cids = _extract_cluster_ids(rt_recs)
        sp_flips = gross_downward_flip_stats(sp_s0, sp_s1, alpha=alpha, cluster_ids=sp_cids)
        rt_flips = gross_downward_flip_stats(rt_s0, rt_s1, alpha=alpha, cluster_ids=rt_cids)
        sp_churn = (
            unbiased_mean_squared_churn(sp_s0, sp_s1, alpha=alpha, cluster_ids=sp_cids)
            if sp_s0.ndim == 2 and sp_s0.shape[1] >= 2 and sp_s1.shape[1] >= 2
            else None
        )
        rt_churn = (
            unbiased_mean_squared_churn(rt_s0, rt_s1, alpha=alpha, cluster_ids=rt_cids)
            if rt_s0.ndim == 2 and rt_s0.shape[1] >= 2 and rt_s1.shape[1] >= 2
            else None
        )
        max_rt_delta = float(th.get("max_routine_false_escalation_delta", 0.080))
        rt_false_esc_delta = float(rt_flips["net_harm_mean"])
        stratum_split = {
            "safety_positive": {
                "n_cases": len(sp_recs),
                "n_clusters": sp_flips["n_clusters"],
                "recall_v0": sp_flips["safety_v0_mean"],
                "recall_v1": sp_flips["safety_v1_mean"],
                "net_recall_delta": sp_flips["net_safety_delta"],
                "psi_down_mean": sp_flips["psi_down_mean"],
                "psi_down_upper": sp_flips["psi_down_upper"],
                "psi_down_upper_unclustered": sp_flips["psi_down_upper_unclustered"],
                "theta_sq_mean": sp_churn["theta_sq_mean"] if sp_churn else None,
            },
            "routine": {
                "n_cases": len(rt_recs),
                "n_clusters": rt_flips["n_clusters"],
                "specificity_v0": rt_flips["safety_v0_mean"],
                "specificity_v1": rt_flips["safety_v1_mean"],
                "false_escalation_delta": rt_false_esc_delta,
                "false_escalation_upper": float(rt_flips["net_harm_upper"]),
                "max_allowed_delta": max_rt_delta,
                "theta_sq_mean": rt_churn["theta_sq_mean"] if rt_churn else None,
            },
        }
        g1_flip_ok = sp_flips["psi_down_upper"] <= float(th["max_gross_flip_upper"])
        g1_routine_ok = rt_false_esc_delta <= max_rt_delta
        # When safety-positive recall rises strictly without excess downward flips,
        # evaluate churn against the routine stratum so beneficial 0->1 fixes do not trip churn.
        g1_churn_ok = (
            True
            if rt_churn is None
            else (
                (overall_churn is not None and overall_churn["theta_sq_mean"] <= float(th["max_churn_upper"]))
                or (g1_flip_ok and sp_flips["net_safety_delta"] >= 0.0 and rt_churn["theta_sq_mean"] <= max_rt_delta)
            )
        )
        flip_n_eff = int(sp_flips["n_clusters"])
        flip_target_records = sp_recs
    else:
        g1_flip_ok = overall_flips["psi_down_upper"] <= float(th["max_gross_flip_upper"])
        g1_routine_ok = True
        g1_churn_ok = True if overall_churn is None else (overall_churn["theta_sq_mean"] <= float(th["max_churn_upper"]))
        flip_n_eff = int(overall_flips["n_clusters"])
        flip_target_records = records

    # Exact power feasibility diagnostic for the active target flip cap
    zero_err_floor = betting_upper_bound(np.zeros(flip_n_eff), alpha=alpha)
    underpowered_for_margin = bool(zero_err_floor > float(th["max_gross_flip_upper"]))

    # Optional profile-product case-mix reweighting across disjoint clinical domains
    domain_reweighted: dict[str, Any] | None = None
    if domain_weights:
        dom_arrays: list[np.ndarray] = []
        dom_w: list[float] = []
        dom_names: list[str] = []
        for dom_name, weight in domain_weights.items():
            d_recs = [r for r in flip_target_records if str(r.get("domain", "")) == dom_name]
            if d_recs and weight > 0:
                dom_arrays.append(_cluster_psi_down_vector(d_recs))
                dom_w.append(float(weight))
                dom_names.append(dom_name)
        if dom_arrays and abs(sum(dom_w) - 1.0) < 1e-6:
            reweighted_mean = float(sum(w_i * float(np.mean(a_i)) for w_i, a_i in zip(dom_w, dom_arrays)))
            reweighted_ub = profile_product_upper_bound(dom_arrays, dom_w, alpha=alpha)
            domain_reweighted = {
                "domains": dom_names,
                "weights": {k: v for k, v in zip(dom_names, dom_w)},
                "cluster_counts": {k: int(a.size) for k, a in zip(dom_names, dom_arrays)},
                "psi_down_mean": reweighted_mean,
                "psi_down_upper": reweighted_ub,
            }

    # Safety e-value for online e-LOND sequential testing (H0: psi_down >= max_gross_flip_upper)
    psi_vec = _cluster_psi_down_vector(flip_target_records)
    safety_evalue = betting_evalue_upper(psi_vec, m=float(th["max_gross_flip_upper"]))

    # One-sided non-inferiority check: H0: net_delta <= margin (equivalently net_harm >= -margin)
    g1_qual_ok = (-q_stats["net_harm_upper"]) >= float(th["net_quality_ni_margin"])
    g1_anchor_ok = True if anchor_res is None else bool(anchor_res["passed"])
    gate1 = {
        "overall": overall_flips,
        "churn": overall_churn,
        "stratum_split": stratum_split,
        "domain_reweighted": domain_reweighted,
        "safety_evalue_at_cap": float(safety_evalue),
        "power_feasibility": {
            "effective_independent_clusters": flip_n_eff,
            "zero_error_betting_floor": float(zero_err_floor),
            "target_max_gross_flip_upper": float(th["max_gross_flip_upper"]),
            "underpowered_for_margin": underpowered_for_margin,
        },
        "quality": {
            "mean_v0": q_stats["safety_v0_mean"],
            "mean_v1": q_stats["safety_v1_mean"],
            "net_delta": q_stats["net_safety_delta"],
            "ni_lower_1s": float(-q_stats["net_harm_upper"]),
            "ci_lower": q_stats["net_safety_ci_lower"],
            "ci_upper": q_stats["net_safety_ci_upper"],
            "passed": g1_qual_ok,
        },
        "anchor": anchor_res,
        "minimum_verifiable_margin": mvm,
        "subgroups": subgroups,
        "checks": {
            "flip_bound_ok": g1_flip_ok,
            "routine_budget_ok": g1_routine_ok,
            "churn_ok": g1_churn_ok,
            "quality_ni_ok": g1_qual_ok,
            "anchor_ok": g1_anchor_ok,
        },
        "passed": bool(g1_flip_ok and g1_routine_ok and g1_churn_ok and g1_qual_ok and g1_anchor_ok),
    }

    # Gate 2: Optional blinded clinician calibration & PPI++ judge shortcut check
    gate2: dict[str, Any] = {"run": False, "passed": True}
    if clinician_calibration_records:
        f0_u = np.array([np.mean(r.get("quality_v0", r["safe_v0"])) for r in records], dtype=float)
        f1_u = np.array([np.mean(r.get("quality_v1", r["safe_v1"])) for r in records], dtype=float)
        f0_l = np.array([float(r["judge_v0"]) for r in clinician_calibration_records], dtype=float)
        f1_l = np.array([float(r["judge_v1"]) for r in clinician_calibration_records], dtype=float)
        y0_l = np.array([float(r["clinician_v0"]) for r in clinician_calibration_records], dtype=float)
        y1_l = np.array([float(r["clinician_v1"]) for r in clinician_calibration_records], dtype=float)
        ppi = ppi_plus_plus_paired(f0_u, f1_u, f0_l, f1_l, y0_l, y1_l, alpha=alpha)
        cal = judge_calibration_equivalence(
            f0_l, f1_l, y0_l, y1_l, tau_cal=float(th["judge_cal_equivalence_margin"]), alpha=alpha
        )
        gate2 = {
            "run": True,
            "ppi": ppi,
            "calibration": cal,
            "passed": bool(cal["calibration_passed"] and not cal["shortcut_alert"]),
        }

    all_passed = bool(gate0["passed"] and gate1["passed"] and gate2["passed"])
    blocking_reasons = []
    if not gate0["passed"]:
        blocking_reasons.append(f"Gate 0 deterministic safety failure ({g0_fail_v1} failed)")
    if not g1_flip_ok:
        if stratum_split is not None:
            sp_ub = stratum_split["safety_positive"]["psi_down_upper"]
            blocking_reasons.append(f"Safety-positive missed-hazard flip upper bound {sp_ub:.3f} > {th['max_gross_flip_upper']:.3f}")
        else:
            blocking_reasons.append(f"Safe-to-unsafe flip upper bound {overall_flips['psi_down_upper']:.3f} > {th['max_gross_flip_upper']:.3f}")
    if not g1_routine_ok and stratum_split is not None:
        rt_d = stratum_split["routine"]["false_escalation_delta"]
        rt_cap = stratum_split["routine"]["max_allowed_delta"]
        blocking_reasons.append(f"Routine false-escalation shift {rt_d:+.3f} > {rt_cap:.3f}")
    if not g1_churn_ok and overall_churn is not None:
        blocking_reasons.append(f"Mean-squared case churn {overall_churn['theta_sq_mean']:.3f} > {th['max_churn_upper']:.3f}")
    if not g1_qual_ok:
        blocking_reasons.append(f"Quality lower bound {q_stats['net_safety_ci_lower']:.3f} < {th['net_quality_ni_margin']:.3f}")
    if not g1_anchor_ok:
        blocking_reasons.append("Cumulative drift vs. locked anchor A0 exceeded limit")
    if gate2["run"] and not gate2["passed"]:
        blocking_reasons.append("Blinded clinician judge calibration failed or shortcut alert triggered")

    report: dict[str, Any] = {
        "schema_version": "0.1.0",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile": prof["profile"],
        "regulatory_notice": prof["regulatory_notice"].strip(),
        "crosswalk": prof["crosswalk"],
        "manifest": manifest,
        "manifest_sha256": hash_manifest(manifest),
        "thresholds": th,
        "decision": "PROMOTED" if all_passed else "BLOCKED",
        "blocking_reasons": blocking_reasons,
        "gate0": gate0,
        "gate1": gate1,
        "gate2": gate2,
    }
    report["bundle_sha256"] = hash_manifest(report)
    return report


def evaluate_release_sequence(
    releases: Sequence[dict[str, Any]],
    profile_name: str = "enterprise_governance",
    alpha_fdr: float = 0.05,
    threshold_overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate an ordered sequence of releases with e-LOND online FDR control (Xu & Ramdas 2024).

    Each item in `releases` must have keys: `release_id`, `records`, `manifest`, and optional
    `clinician_calibration_records` and `domain_weights`.
    At step t, tests H0_t: psi_down >= max_gross_flip_upper using the exact cluster-aware
    safety e-value; certifying safety requires both standard gates AND e-LOND rejection of H0_t.
    """
    elond_state: dict[str, int] = {"k": 0, "rejections": 0}
    steps: list[dict[str, Any]] = []
    for item in releases:
        rep = evaluate_release(
            records=item["records"],
            manifest=item["manifest"],
            profile_name=profile_name,
            clinician_calibration_records=item.get("clinician_calibration_records"),
            threshold_overrides=threshold_overrides,
            domain_weights=item.get("domain_weights"),
        )
        e_val = float(rep["gate1"]["safety_evalue_at_cap"])
        elond_res = elond_step(e_val, alpha=alpha_fdr, state=elond_state)
        online_promoted = bool(
            rep["gate0"]["passed"]
            and rep["gate1"]["checks"]["routine_budget_ok"]
            and rep["gate1"]["checks"]["churn_ok"]
            and rep["gate1"]["checks"]["quality_ni_ok"]
            and rep["gate1"]["checks"]["anchor_ok"]
            and rep["gate2"]["passed"]
            and elond_res["rejected"]
        )
        steps.append({
            "release_id": str(item["release_id"]),
            "single_release_decision": rep["decision"],
            "elond_decision": "PROMOTED" if online_promoted else "BLOCKED",
            "elond": elond_res,
            "gate0_failures_v1": rep["gate0"]["failures_v1"],
            "psi_down_mean": (
                rep["gate1"]["stratum_split"]["safety_positive"]["psi_down_mean"]
                if rep["gate1"]["stratum_split"]
                else rep["gate1"]["overall"]["psi_down_mean"]
            ),
            "psi_down_upper": (
                rep["gate1"]["stratum_split"]["safety_positive"]["psi_down_upper"]
                if rep["gate1"]["stratum_split"]
                else rep["gate1"]["overall"]["psi_down_upper"]
            ),
            "blocking_reasons": rep["blocking_reasons"],
        })
    return {
        "profile": profile_name,
        "alpha_fdr": float(alpha_fdr),
        "n_releases": len(steps),
        "n_single_promoted": sum(1 for s in steps if s["single_release_decision"] == "PROMOTED"),
        "n_elond_promoted": sum(1 for s in steps if s["elond_decision"] == "PROMOTED"),
        "steps": steps,
    }


def verify_bundle_integrity(report: dict[str, Any]) -> bool:
    """Recompute SHA-256 over the report excluding bundle_sha256 and check exact match."""
    claimed = report.get("bundle_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        return False
    payload = {k: v for k, v in report.items() if k != "bundle_sha256"}
    return hash_manifest(payload) == claimed
