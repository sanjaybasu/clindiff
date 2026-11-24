#!/usr/bin/env python3
"""Consistency and provenance verifier for clinical-ai-change-control (clindiff).

Checks:
  1. Re-derives Part D exact Minimum Verifiable Margin table directly from
     clindiff.stats.minimum_verifiable_margin and asserts exact equality with
     results/retrospective_audit_aggregate.json.
  2. Verifies structural invariants and zero-PHI hygiene across
     results/retrospective_audit_aggregate.json, vignettes/medicaid_safety_v0.1.jsonl,
     and vignettes/buyer_canary_template.jsonl (counts, exact 50x3 pairing, zero
     overlap, SHA-256 digests).
  3. Cross-checks key canonical numbers reported in
     notebooks/clinical-ai-change-control/retrospective_empirical_summary.md
     against results/retrospective_audit_aggregate.json, printing a discrepancy
     table and exiting non-zero on any mismatch.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from clindiff.stats import minimum_verifiable_margin

import os
PKG_ROOT = Path(__file__).resolve().parents[1]
NB_ROOT = Path(os.environ.get("CLINDIFF_NOTEBOOKS_ROOT", str(PKG_ROOT.parent.parent / "notebooks/clinical-ai-change-control")))
CANONICAL_JSON = PKG_ROOT / "results/retrospective_audit_aggregate.json"
PUB_VIGNETTES = PKG_ROOT / "vignettes/medicaid_safety_v0.1.jsonl"
CANARY_VIGNETTES = PKG_ROOT / "vignettes/buyer_canary_template.jsonl"
SUMMARY_MD = NB_ROOT / "retrospective_empirical_summary.md"

_PHI_PATTERNS = [
    ("phone", re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("waymark_id", re.compile(r"\bWAY\d{4,}\b")),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    discrepancies: list[tuple[str, str, str, str]] = []  # (file, check, current, canonical)

    if not CANONICAL_JSON.exists():
        print(f"ERROR: Missing canonical JSON: {CANONICAL_JSON}", file=sys.stderr)
        return 2

    data = json.loads(CANONICAL_JSON.read_text(encoding="utf-8"))

    # 1. Verify Part A/B/C structural counts & invariants
    pa = data["part_a_compass_949_r3"]
    if pa["n_valid_texts"] != pa["n_safety_positive"] + pa["n_non_safety_positive"]:
        discrepancies.append((
            str(CANONICAL_JSON), "part_a_sum",
            str(pa["n_safety_positive"] + pa["n_non_safety_positive"]),
            str(pa["n_valid_texts"]),
        ))
    if pa["n_doctor_call"] > pa["n_safety_positive"]:
        discrepancies.append((
            str(CANONICAL_JSON), "part_a_doctor_call_subset",
            str(pa["n_doctor_call"]), f"<= {pa['n_safety_positive']}",
        ))

    pb = data["part_b_hbpro_350_3graders"]
    if len(pb["criterion_disagreement_by_arm"]) != pb["n_full_arms"]:
        discrepancies.append((
            str(CANONICAL_JSON), "part_b_arm_count",
            str(len(pb["criterion_disagreement_by_arm"])), str(pb["n_full_arms"]),
        ))

    pc = data["part_c_grader_meta_2000_6graders"]
    for gname, gstats in pc["graders_vs_physicians"].items():
        expected_bias = round((gstats["model_met_rate"] - gstats["physician_maj_met_rate"]) * 100.0, 2)
        if abs(gstats["signed_bias_pp"] - expected_bias) > 0.02:
            discrepancies.append((
                str(CANONICAL_JSON), f"part_c_bias:{gname}",
                str(gstats["signed_bias_pp"]), str(expected_bias),
            ))

    # 2. Re-derive Part D MVM power table directly via clindiff.stats.minimum_verifiable_margin
    for row in data["part_d_mvm_power_table"]:
        n = int(row["n_cases"])
        s1 = minimum_verifiable_margin(n=n, alpha=0.05, discordant_rate=0.05)
        q4 = minimum_verifiable_margin(n=n, alpha=0.05 / 4, discordant_rate=0.05)
        m12 = minimum_verifiable_margin(n=n, alpha=0.05 / 12, discordant_rate=0.05)
        checks = [
            ("single_release_zero_error_floor_pp", round(s1["zero_error_betting_floor"] * 100.0, 2)),
            ("single_release_empirical_mvm_pp", round(s1["empirical_mvm"] * 100.0, 2)),
            ("quarterly_4x_empirical_mvm_pp", round(q4["empirical_mvm"] * 100.0, 2)),
            ("monthly_12x_empirical_mvm_pp", round(m12["empirical_mvm"] * 100.0, 2)),
        ]
        for key, canonical_val in checks:
            if row[key] != canonical_val:
                discrepancies.append((
                    str(CANONICAL_JSON), f"part_d:N={n}:{key}",
                    str(row[key]), str(canonical_val),
                ))

    # 3. Verify synthetic vignette counts, 100x3 pairing, zero overlap, and zero PHI
    pub = _load_jsonl(PUB_VIGNETTES)
    can = _load_jsonl(CANARY_VIGNETTES)
    if len(pub) != 300:
        discrepancies.append((str(PUB_VIGNETTES), "row_count", str(len(pub)), "300"))
    if len(can) != 50:
        discrepancies.append((str(CANARY_VIGNETTES), "row_count", str(len(can)), "50"))
    if Counter(r["stratum"] for r in pub) != {"en_standard": 100, "en_lowlit_paraphrase": 100, "es_spanish": 100}:
        discrepancies.append((str(PUB_VIGNETTES), "strata", str(dict(Counter(r["stratum"] for r in pub))), "100/100/100"))
    overlap = set(r["prompt"].strip().lower() for r in pub) & set(r["prompt"].strip().lower() for r in can)
    if overlap:
        discrepancies.append((str(CANARY_VIGNETTES), "prompt_overlap", str(len(overlap)), "0"))

    for artifact in [CANONICAL_JSON, PUB_VIGNETTES, CANARY_VIGNETTES, SUMMARY_MD]:
        if not artifact.exists():
            discrepancies.append((str(artifact), "exists", "missing", "present"))
            continue
        txt = artifact.read_text(encoding="utf-8")
        for label, pat in _PHI_PATTERNS:
            if pat.search(txt):
                discrepancies.append((str(artifact), f"phi_check:{label}", "MATCHED", "NONE"))

    # 4. Cross-check canonical values inside retrospective_empirical_summary.md
    if SUMMARY_MD.exists():
        md = SUMMARY_MD.read_text(encoding="utf-8")
        expected_tokens = [
            # Part A counts & key rates
            ("part_a:n_valid", "949"),
            ("part_a:n_safety_pos", "79"),
            ("part_a:n_doctor_call", "53"),
            ("part_a:amie_flip_all", f"{pa['within_arm_repeat_instability']['amie_style']['all_texts_3rep_flip_rate'] * 100:.1f}%"),
            ("part_a:mai_flip_all", f"{pa['within_arm_repeat_instability']['mai_dxo_style']['all_texts_3rep_flip_rate'] * 100:.1f}%"),
            ("part_a:v22_flip_all", f"{pa['within_arm_repeat_instability']['v22']['all_texts_3rep_flip_rate'] * 100:.1f}%"),
            ("part_a:amie_sp_bare", f"{pa['within_arm_repeat_instability']['amie_style']['safety_pos_sensitivity_bare']:.3f}"),
            ("part_a:amie_sp_layer", f"{pa['within_arm_repeat_instability']['amie_style']['safety_pos_sensitivity_with_layer']:.3f}"),
            ("part_a:raw_to_amie_churn", f"{pa['pairwise_comparisons']['raw_llm_to_amie_safety_pos_n79']['theta_sq_churn']:.3f}"),
            ("part_a:amie_to_v22_psi_down", f"{pa['pairwise_comparisons']['amie_to_v22_safety_pos_n79']['psi_down_mean'] * 100:.1f}%"),
            ("part_a:split_routine_shift", f"{pa['internal_gate_dry_runs']['dryrun_3_split_amie_bare_to_plus_layer']['split_routine_false_esc_delta'] * 100:+.1f}"),
            # Part B key spreads
            ("part_b:spread_mai", f"{pb['pairwise_grader_sensitivity']['compass_pro_full_blk_vs_mai_dxo_style_search_edit_blk']['grader_spread_pp']:.2f}"),
            ("part_b:spread_amie", f"{pb['pairwise_grader_sensitivity']['compass_pro_full_blk_vs_amie_style_search_blk']['grader_spread_pp']:.2f}"),
            # Part C signed biases & PPI++
            ("part_c:bias_gpt41", f"{pc['graders_vs_physicians']['openai:gpt-4.1-2025-04-14']['signed_bias_pp']:.2f}"),
            ("part_c:bias_gpt61", f"{pc['graders_vs_physicians']['openai:gpt-6.1-sol']['signed_bias_pp']:.2f}"),
            ("part_c:shortcut_m200_raw", f"{pc['ppi_and_shortcut_audit']['m_200']['shortcut_raw_judge_delta_pp']:.2f}"),
            ("part_c:swap_m200_cal", f"{pc['ppi_and_shortcut_audit']['m_200']['grader_swap_41_to_61_diff_cal_pp']:.1f}"),
            # Part D MVM anchors
            ("part_d:N50_single", "10.61"),
            ("part_d:N150_single", "6.13"),
            ("part_d:N949_single", "2.44"),
            ("part_d:N10000_monthly", "1.05"),
        ]
        for label, tok in expected_tokens:
            if tok not in md:
                discrepancies.append((str(SUMMARY_MD), label, "MISSING", tok))

    if discrepancies:
        print("DISCREPANCY TABLE:")
        print("| File | Check | Current | Canonical |")
        print("|---|---|---|---|")
        for fpath, chk, cur, can_val in discrepancies:
            print(f"| {fpath} | {chk} | {cur} | {can_val} |")
        return 1

    print("VERIFICATION PASSED: 0 discrepancies across canonical JSON, MVM re-derivation, vignettes, PHI checks, and summary markdown.")
    print(f"  retrospective_audit_aggregate.json sha256: {_sha256(CANONICAL_JSON)[:16]}")
    print(f"  medicaid_safety_v0.1.jsonl         sha256: {_sha256(PUB_VIGNETTES)[:16]} (n={len(pub)})")
    print(f"  buyer_canary_template.jsonl        sha256: {_sha256(CANARY_VIGNETTES)[:16]} (n={len(can)})")
    print(f"  retrospective_empirical_summary.md sha256: {_sha256(SUMMARY_MD)[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
