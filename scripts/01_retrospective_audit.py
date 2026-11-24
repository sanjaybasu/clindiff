#!/usr/bin/env python3
"""Experiment 1 ($0 API cash): Retrospective empirical audit of clindiff primitives on saved runs.

Analyzes four saved clinical datasets (aggregate statistics ONLY; zero PHI or row IDs exported):
  A. Compass 949 physician-labeled Medicaid texts x 3 independent repeats across 4 reply arms
     + separated Kev-9B safety layer (internal dry run of clindiff gates).
  B. HealthBench Professional 350 test tasks (774 rubric items) x 3 frontier graders
     (GPT-5.4, GPT-6.1 Sol, Claude Opus 5.5) across 10 arms (grader vs. case variance).
  C. HealthBench Meta-Eval 2,000 items x 6 frontier graders vs. physicians
     (predictable PPI++ and fail-closed differential judge-calibration audit).
  D. Exact Minimum Verifiable Margin (Delta_min) power table across N = 30 to 25,000.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl

from clindiff.stats import (
    gross_downward_flip_stats,
    unbiased_mean_squared_churn,
    minimum_verifiable_margin,
    ppi_plus_plus_paired,
    judge_calibration_equivalence,
)
from clindiff.gates import evaluate_release

PKG_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("CLINDIFF_DATA_ROOT", str(PKG_ROOT.parent / "compass-pcp-data")))
HBPRO_JSONL = Path.home() / (
    ".cache/huggingface/hub/datasets--openai--healthbench-professional/snapshots/"
    "349962fd46dd02343a0d8a606491baf59154ea1a/healthbench_professional_eval.jsonl"
)
META_JSONL = sorted(
    (Path.home() / ".cache/huggingface/hub/datasets--openai--healthbench/snapshots").glob(
        "*/2025-05-07-06-14-12_oss_meta_eval.jsonl"
    )
)[-1]
SIMPLE_EVALS_PARENT = DATA_ROOT / "vendor/se"
OUT_JSON = PKG_ROOT / "results/retrospective_audit_aggregate.json"


def _load_compass_labels() -> dict[str, dict[str, Any]]:
    """Load physician hazard/action labels mapped via local crosswalk (never exported)."""
    xwalk: dict[str, str] = {}
    with (DATA_ROOT / "fresh_set/crosswalk.jsonl").open() as f:
        for line in f:
            o = json.loads(line)
            xwalk[str(o["sample_id"])] = str(o["id"])

    wb = openpyxl.load_workbook(DATA_ROOT / "fresh_set/labels/basu_completed.xlsx", read_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()
    hdr = [str(c).strip() if c is not None else "" for c in rows[0]]
    idx_id = hdr.index("sample_id")
    idx_haz = hdr.index("hazard (0/1)")
    idx_cat = hdr.index("hazard_category")
    idx_act = hdr.index("action")
    out: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        sid = row[idx_id]
        if not sid or str(sid) not in xwalk:
            continue
        haz = int(row[idx_haz]) if row[idx_haz] is not None else 0
        cat = str(row[idx_cat] or "").strip().lower()
        act = str(row[idx_act] or "").strip().lower()
        # Matches compass-pcp-paper2 canonical definitions:
        # safety_positive (n=79): hazard==1 excluding routine 'other_hazard'
        # doctor_call (n=53): action in {'Contact Doctor', 'Call 911/988'}
        safety_positive = (haz == 1) and (cat != "other_hazard")
        doctor_call = ("911" in act) or ("988" in act) or ("contact doctor" in act)
        out[xwalk[str(sid)]] = {
            "any_hazard": haz == 1,
            "safety_positive": safety_positive,
            "doctor_call": doctor_call,
        }
    return out


def run_part_a_compass_texts() -> dict[str, Any]:
    """Part A: 949 Medicaid texts x R=3 repeats across 4 arms + Kev-9B safety layer."""
    labels = _load_compass_labels()
    gate_th = json.loads((DATA_ROOT / "paper2_final/texts/gate_thresholds.json").read_text())
    t_e, t_s = float(gate_th["t_e"]), float(gate_th["t_s"])
    gate_flagged: dict[str, bool] = {}
    with (DATA_ROOT / "paper2_final/texts/gate_scores.jsonl").open() as f:
        for line in f:
            o = json.loads(line)
            gate_flagged[str(o["id"])] = (float(o["emergency"]) >= t_e) or (float(o["safety"]) >= t_s)

    # Group urgency by (arm, prompt_id) -> dict[repeat -> escalated_bool]
    # Escalated = anything other than 'none' (matches compass-pcp-paper2 definition)
    arm_rep: dict[str, dict[str, dict[int, int]]] = defaultdict(lambda: defaultdict(dict))
    with (DATA_ROOT / "paper2_final/texts/urgency.jsonl").open() as f:
        for line in f:
            o = json.loads(line)
            pid = str(o["prompt_id"])
            if pid not in labels:
                continue
            arm = str(o["arm"])
            rep = int(o["repeat"])
            esc = 0 if str(o.get("urgency", "none")) == "none" else 1
            arm_rep[arm][pid][rep] = esc

    arms = sorted(arm_rep.keys())
    common_ids = sorted(set.intersection(*(set(arm_rep[a].keys()) for a in arms)))
    # Keep only ids with all 3 repeats present across all 4 arms
    valid_ids = [
        pid for pid in common_ids
        if all(all(r in arm_rep[a][pid] for r in (0, 1, 2)) for a in arms)
    ]
    sp_ids = [pid for pid in valid_ids if labels[pid]["safety_positive"]]
    dc_ids = [pid for pid in valid_ids if labels[pid]["doctor_call"]]
    non_sp_ids = [pid for pid in valid_ids if not labels[pid]["safety_positive"]]

    def _matrix(arm: str, ids: list[str], with_layer: bool = False, score_correct_action: bool = False) -> np.ndarray:
        mat = np.zeros((len(ids), 3), dtype=float)
        for i, pid in enumerate(ids):
            flagged = gate_flagged.get(pid, False)
            is_sp = labels[pid]["safety_positive"]
            for r in range(3):
                esc = arm_rep[arm][pid][r]
                if with_layer and flagged:
                    esc = 1
                if score_correct_action:
                    mat[i, r] = float(esc == (1 if is_sp else 0))
                else:
                    mat[i, r] = float(esc)
        return mat

    # 1. Within-arm repeat instability (R=3 identical repeats on same patient text)
    within_arm = {}
    for arm in arms:
        m_all = _matrix(arm, valid_ids, with_layer=False)
        m_sp = _matrix(arm, sp_ids, with_layer=False)
        m_sp_layer = _matrix(arm, sp_ids, with_layer=True)
        m_dc = _matrix(arm, dc_ids, with_layer=False)
        m_dc_layer = _matrix(arm, dc_ids, with_layer=True)
        any_flip_all = float(np.mean((m_all.sum(axis=1) > 0) & (m_all.sum(axis=1) < 3)))
        any_flip_sp = float(np.mean((m_sp.sum(axis=1) > 0) & (m_sp.sum(axis=1) < 3)))
        any_flip_sp_layer = float(np.mean((m_sp_layer.sum(axis=1) > 0) & (m_sp_layer.sum(axis=1) < 3)))
        any_flip_dc = float(np.mean((m_dc.sum(axis=1) > 0) & (m_dc.sum(axis=1) < 3)))
        any_flip_dc_layer = float(np.mean((m_dc_layer.sum(axis=1) > 0) & (m_dc_layer.sum(axis=1) < 3)))
        st_all = gross_downward_flip_stats(m_all, m_all, alpha=0.05)
        st_sp = gross_downward_flip_stats(m_sp, m_sp, alpha=0.05)
        st_sp_l = gross_downward_flip_stats(m_sp_layer, m_sp_layer, alpha=0.05)
        within_arm[arm] = {
            "n_all": len(valid_ids),
            "n_safety_positive": len(sp_ids),
            "n_doctor_call": len(dc_ids),
            "all_texts_mean_escalation": round(float(m_all.mean()), 4),
            "all_texts_3rep_flip_rate": round(any_flip_all, 4),
            "all_texts_psi_self": round(st_all["psi_self_mean"], 4),
            "safety_pos_sensitivity_bare": round(float(m_sp.mean()), 4),
            "safety_pos_3rep_flip_rate_bare": round(any_flip_sp, 4),
            "safety_pos_psi_self_bare": round(st_sp["psi_self_mean"], 4),
            "safety_pos_sensitivity_with_layer": round(float(m_sp_layer.mean()), 4),
            "safety_pos_3rep_flip_rate_with_layer": round(any_flip_sp_layer, 4),
            "safety_pos_psi_self_with_layer": round(st_sp_l["psi_self_mean"], 4),
            "doctor_call_sensitivity_bare": round(float(m_dc.mean()), 4),
            "doctor_call_3rep_flip_rate_bare": round(any_flip_dc, 4),
            "doctor_call_sensitivity_with_layer": round(float(m_dc_layer.mean()), 4),
            "doctor_call_3rep_flip_rate_with_layer": round(any_flip_dc_layer, 4),
        }

    # 2. Pairwise architecture / update comparisons on safety-positive (n=79), doctor-call (n=53), & full 949
    comparisons = [
        ("raw_llm_to_amie_safety_pos_n79", "raw_llm", False, "amie_style", False, sp_ids, False),
        ("raw_llm_to_mai_dxo_safety_pos_n79", "raw_llm", False, "mai_dxo_style", False, sp_ids, False),
        ("amie_to_v22_safety_pos_n79", "amie_style", False, "v22", False, sp_ids, False),
        ("amie_bare_to_plus_layer_safety_pos_n79", "amie_style", False, "amie_style", True, sp_ids, False),
        ("amie_bare_to_plus_layer_doctor_call_n53", "amie_style", False, "amie_style", True, dc_ids, False),
        ("raw_llm_to_amie_all949_accuracy", "raw_llm", False, "amie_style", False, valid_ids, True),
        ("amie_to_mai_dxo_all949_accuracy", "amie_style", False, "mai_dxo_style", False, valid_ids, True),
    ]
    pairwise = {}
    for name, a0, l0, a1, l1, ids, score_act in comparisons:
        m0 = _matrix(a0, ids, with_layer=l0, score_correct_action=score_act)
        m1 = _matrix(a1, ids, with_layer=l1, score_correct_action=score_act)
        fl = gross_downward_flip_stats(m0, m1, alpha=0.05)
        ch = unbiased_mean_squared_churn(m0, m1, alpha=0.05)
        pairwise[name] = {
            "n_cases": len(ids),
            "v0_mean": round(fl["safety_v0_mean"], 4),
            "v1_mean": round(fl["safety_v1_mean"], 4),
            "net_delta": round(fl["net_safety_delta"], 4),
            "psi_down_mean": round(fl["psi_down_mean"], 4),
            "psi_down_upper_95": round(fl["psi_down_upper"], 4),
            "psi_self_v0": round(fl["psi_self_mean"], 4),
            "theta_sq_churn": round(ch["theta_sq_mean"], 4),
            "theta_sq_ci_95": [round(ch["theta_sq_lower"], 4), round(ch["theta_sq_upper_2s"], 4)],
        }

    # 3. Internal dry run of clindiff.gates.evaluate_release on realistic release transitions
    def _gate_dry_run(a0: str, l0: bool, a1: str, l1: bool, label: str, *, split_strata: bool = False) -> dict[str, Any]:
        m0 = _matrix(a0, valid_ids, with_layer=l0, score_correct_action=True)
        m1 = _matrix(a1, valid_ids, with_layer=l1, score_correct_action=True)
        recs = []
        for i, pid in enumerate(valid_ids):
            rec = {
                "case_id": f"anon_{i:04d}",
                "stratum": "safety_positive" if labels[pid]["safety_positive"] else "non_safety_positive",
                "gate0_v0_pass": True,
                "gate0_v1_pass": True,
                "safe_v0": m0[i].tolist(),
                "safe_v1": m1[i].tolist(),
                "safe_anchor": m0[i].tolist(),
                "quality_v0": m0[i].tolist(),
                "quality_v1": m1[i].tolist(),
            }
            if split_strata:
                rec["is_safety_positive"] = bool(labels[pid]["safety_positive"])
            recs.append(rec)
        rep = evaluate_release(
            records=recs,
            manifest={"system_name": "Compass-Internal-DryRun", "transition": label, "n_texts": len(valid_ids)},
            profile_name="enterprise_governance",
        )
        out = {
            "decision": rep["decision"],
            "blocking_reasons": rep["blocking_reasons"],
            "mvm_empirical": round(rep["gate1"]["minimum_verifiable_margin"]["empirical_mvm"], 4),
            "psi_down_mean": round(rep["gate1"]["overall"]["psi_down_mean"], 4),
            "psi_down_upper": round(rep["gate1"]["overall"]["psi_down_upper"], 4),
            "theta_sq_mean": round(rep["gate1"]["churn"]["theta_sq_mean"], 4),
            "safety_pos_stratum_psi_down": round(rep["gate1"]["subgroups"]["safety_positive"]["psi_down_mean"], 4),
            "safety_pos_stratum_delta": round(rep["gate1"]["subgroups"]["safety_positive"]["net_safety_delta"], 4),
        }
        if split_strata and rep["gate1"]["stratum_split"] is not None:
            sp = rep["gate1"]["stratum_split"]["safety_positive"]
            rt = rep["gate1"]["stratum_split"]["routine"]
            out["split_safety_pos_psi_down_upper"] = round(sp["psi_down_upper"], 4)
            out["split_routine_false_esc_delta"] = round(rt["false_escalation_delta"], 4)
            out["split_routine_budget_ok"] = bool(rep["gate1"]["checks"]["routine_budget_ok"])
        return out

    dry_runs = {
        "dryrun_1_raw_to_amie": _gate_dry_run("raw_llm", False, "amie_style", False, "raw_llm->amie_style"),
        "dryrun_2_amie_to_mai_dxo": _gate_dry_run("amie_style", False, "mai_dxo_style", False, "amie_style->mai_dxo_style"),
        "dryrun_3_amie_bare_to_plus_layer": _gate_dry_run("amie_style", False, "amie_style", True, "amie->amie+layer"),
        "dryrun_3_split_amie_bare_to_plus_layer": _gate_dry_run("amie_style", False, "amie_style", True, "amie->amie+layer (split)", split_strata=True),
    }

    return {
        "n_valid_texts": len(valid_ids),
        "n_safety_positive": len(sp_ids),
        "n_doctor_call": len(dc_ids),
        "n_non_safety_positive": len(non_sp_ids),
        "within_arm_repeat_instability": within_arm,
        "pairwise_comparisons": pairwise,
        "internal_gate_dry_runs": dry_runs,
    }


def run_part_b_hbpro_graders() -> dict[str, Any]:
    """Part B: HealthBench Professional 350 test tasks x 3 frontier graders."""
    # Load rubric point weights per (id, k)
    rubric_pts: dict[str, list[float]] = {}
    with HBPRO_JSONL.open() as f:
        for line in f:
            o = json.loads(line)
            pid = str(o.get("id") or o.get("prompt_id"))
            rubrics = o.get("rubric_items", [])
            rubric_pts[pid] = [float(r["points"]) for r in rubrics]

    grader_files = {
        "gpt54": DATA_ROOT / "hbpro_test/grades.jsonl",
        "gpt61sol": DATA_ROOT / "hbpro_test/grades_g61.jsonl",
        "opus55": DATA_ROOT / "hbpro_test/grades_opus.jsonl",
    }
    # Load grades[(grader, arm, id, k)] = bool
    grades: dict[str, dict[str, dict[tuple[str, int], int]]] = defaultdict(lambda: defaultdict(dict))
    for gname, path in grader_files.items():
        with path.open() as f:
            for line in f:
                o = json.loads(line)
                grades[gname][str(o["arm"])][(str(o["id"]), int(o["k"]))] = 1 if o["met"] else 0

    shared_arms = sorted(set.intersection(*(set(grades[g].keys()) for g in grader_files)))
    # Restrict to arms with full 774 rubric items across all 3 graders
    full_arms = [
        a for a in shared_arms
        if all(len(grades[g][a]) == 774 for g in grader_files)
    ]

    # 1. Criterion-level grader disagreement across the 774 binary rubric items per arm
    criterion_disagreement = {}
    for arm in full_arms:
        keys = sorted(grades["gpt54"][arm].keys())
        mat = np.array([[grades[g][arm][k] for g in ("gpt54", "gpt61sol", "opus55")] for k in keys], dtype=float)
        # Disagreement: not unanimous across the 3 frontier graders
        non_unanimous = float(np.mean((mat.sum(axis=1) > 0) & (mat.sum(axis=1) < 3)))
        # Pairwise disagreement between GPT-5.4 and GPT-6.1 Sol, GPT-5.4 and Opus 5.5
        d_54_61 = float(np.mean(mat[:, 0] != mat[:, 1]))
        d_54_op = float(np.mean(mat[:, 0] != mat[:, 2]))
        d_61_op = float(np.mean(mat[:, 1] != mat[:, 2]))
        criterion_disagreement[arm] = {
            "n_rubric_items": len(keys),
            "met_rate_gpt54": round(float(mat[:, 0].mean()), 4),
            "met_rate_gpt61sol": round(float(mat[:, 1].mean()), 4),
            "met_rate_opus55": round(float(mat[:, 2].mean()), 4),
            "non_unanimous_across_3_graders": round(non_unanimous, 4),
            "pairwise_flip_gpt54_vs_gpt61sol": round(d_54_61, 4),
            "pairwise_flip_gpt54_vs_opus55": round(d_54_op, 4),
            "pairwise_flip_gpt61sol_vs_opus55": round(d_61_op, 4),
        }

    # 2. Task-level unweighted rubric score & variance decomposition (between-task vs grader)
    # Using exact task rubric weights when id matches HF file
    task_ids = sorted({k[0] for k in grades["gpt54"][full_arms[0]].keys()})
    matched_pts = sum(1 for tid in task_ids if tid in rubric_pts)

    def _task_scores(gname: str, arm: str) -> np.ndarray:
        by_tid: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for (tid, k), met in grades[gname][arm].items():
            by_tid[tid].append((k, met))
        scores = []
        for tid in task_ids:
            items = by_tid[tid]
            if tid in rubric_pts:
                pts = rubric_pts[tid]
                pos_total = sum(p for p in pts if p > 0)
                earned = sum(pts[k] for k, met in items if met and k < len(pts))
                s = max(0.0, min(1.0, earned / pos_total)) if pos_total > 0 else 0.0
            else:
                s = float(np.mean([met for _, met in items]))
            scores.append(s)
        return np.array(scores, dtype=float)

    # Key matched-condition pairwise comparisons across all 3 graders
    key_pairs = [
        ("compass_pro_full_blk", "amie_style_search_blk"),
        ("compass_pro_full_blk", "mai_dxo_style_search_edit_blk"),
        ("opus_search_edit_blk", "amie_style_search_blk"),
        ("amie_style", "mai_dxo_style"),
    ]
    grader_sensitivity = {}
    for a, b in key_pairs:
        if a not in full_arms or b not in full_arms:
            continue
        diffs_by_g = {}
        winner_by_g = {}
        for g in ("gpt54", "gpt61sol", "opus55"):
            sa = _task_scores(g, a)
            sb = _task_scores(g, b)
            d = (sa - sb) * 100.0
            diffs_by_g[g] = round(float(np.mean(d)), 2)
            # Per-task winner (+1 a wins, -1 b wins, 0 tie)
            winner_by_g[g] = np.sign(sa - sb)
        # Task-level winner inversion rate between GPT-5.4 and GPT-6.1 Sol (strict wins flipping sign)
        strict_flip_54_61 = float(np.mean((winner_by_g["gpt54"] * winner_by_g["gpt61sol"]) < 0))
        strict_flip_54_op = float(np.mean((winner_by_g["gpt54"] * winner_by_g["opus55"]) < 0))
        any_verdict_change_54_61 = float(np.mean(winner_by_g["gpt54"] != winner_by_g["gpt61sol"]))
        # Variance decomposition across graders for arm difference
        d_mat = np.vstack([(_task_scores(g, a) - _task_scores(g, b)) * 100.0 for g in ("gpt54", "gpt61sol", "opus55")])  # (3, n_tasks)
        within_task_grader_var = float(np.mean(np.var(d_mat, axis=0, ddof=1)))
        between_task_var = float(np.var(np.mean(d_mat, axis=0), ddof=1))
        grader_sensitivity[f"{a}_vs_{b}"] = {
            "n_tasks": len(task_ids),
            "mean_diff_pp_by_grader": diffs_by_g,
            "grader_spread_pp": round(max(diffs_by_g.values()) - min(diffs_by_g.values()), 2),
            "task_strict_winner_flip_gpt54_vs_gpt61": round(strict_flip_54_61, 4),
            "task_strict_winner_flip_gpt54_vs_opus55": round(strict_flip_54_op, 4),
            "task_any_verdict_change_gpt54_vs_gpt61": round(any_verdict_change_54_61, 4),
            "within_task_grader_sd_pp": round(float(np.sqrt(within_task_grader_var)), 2),
            "between_task_sd_pp": round(float(np.sqrt(between_task_var)), 2),
        }

    return {
        "n_full_arms": len(full_arms),
        "n_tasks": len(task_ids),
        "matched_rubric_weight_tasks": matched_pts,
        "criterion_disagreement_by_arm": criterion_disagreement,
        "pairwise_grader_sensitivity": grader_sensitivity,
    }


def _load_grader_template() -> str:
    """Extract GRADER_TEMPLATE string literal via AST without importing blobfile."""
    import ast
    src = (SIMPLE_EVALS_PARENT / "simple_evals/healthbench_eval.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "GRADER_TEMPLATE":
                    # Handles """...""".strip()
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                        return ast.literal_eval(node.value.func.value).strip()
                    return ast.literal_eval(node.value)
    raise RuntimeError("GRADER_TEMPLATE not found")


def run_part_c_grader_meta_ppi() -> dict[str, Any]:
    """Part C: 2,000 HealthBench meta-eval items x 6 frontier graders vs. real physicians."""
    GRADER_TEMPLATE = _load_grader_template()

    # Reconstruct exact prompt SHA-256 -> physician majority label mapping from local HF cache
    phys_by_sha: dict[str, dict[str, Any]] = {}
    with META_JSONL.open() as f:
        for line in f:
            o = json.loads(line)
            convo_str = "\n\n".join(f"{m['role']}: {m['content']}" for m in o["prompt"]) + f"\n\nassistant: {o['completion']}"
            prompt_content = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", o["rubric"])
            sha = hashlib.sha256(prompt_content.encode()).hexdigest()
            bl = [bool(x) for x in o["binary_labels"]]
            phys_by_sha[sha] = {
                "phys_mean": float(np.mean(bl)),
                "phys_maj": 1.0 if sum(bl) >= len(bl) / 2.0 else 0.0,
                "n_phys": len(bl),
                "unanimous": all(bl) or not any(bl),
                "category": str(o.get("category", "unknown")),
            }

    grader_files = sorted((DATA_ROOT / "grader_meta").glob("*.json"))
    grader_maps: dict[str, dict[str, float]] = {}
    summary_by_grader: dict[str, Any] = {}
    for p in grader_files:
        rec = json.loads(p.read_text())
        gname = str(rec["grader"])
        lmap = {k: (1.0 if v else 0.0) for k, v in rec["labels_by_prompt_sha256"].items() if k in phys_by_sha}
        grader_maps[gname] = lmap
        keys = sorted(lmap.keys())
        y_phys = np.array([phys_by_sha[k]["phys_maj"] for k in keys], dtype=float)
        f_mod = np.array([lmap[k] for k in keys], dtype=float)
        # Signed bias f - y (positive = lenient, negative = overly strict)
        signed_bias = float(np.mean(f_mod - y_phys))
        raw_acc = float(np.mean(f_mod == y_phys))
        unanim_mask = np.array([phys_by_sha[k]["unanimous"] for k in keys], dtype=bool)
        acc_unanim = float(np.mean(f_mod[unanim_mask] == y_phys[unanim_mask])) if unanim_mask.any() else float("nan")
        summary_by_grader[gname] = {
            "n_matched": len(keys),
            "model_met_rate": round(float(f_mod.mean()), 4),
            "physician_maj_met_rate": round(float(y_phys.mean()), 4),
            "signed_bias_pp": round(signed_bias * 100.0, 2),
            "raw_agreement_vs_phys_maj": round(raw_acc, 4),
            "agreement_on_unanimous_phys_items": round(acc_unanim, 4),
            "saved_model_f1_balanced": round(float(rec["model_f1_balanced"]), 4),
        }

    # Simulate realistic judge-swap / shortcut test on real physician-labeled items:
    # Compare GPT-4.1 (calibrated) vs GPT-6.1-sol (strict reasoning model) on identical items
    # AND inject a +6 pp verbosity shortcut into GPT-4.1 to verify clindiff PPI++ & calibration gate catch it.
    g_ref = "openai:gpt-4.1-2025-04-14"
    g_strict = "openai:gpt-6.1-sol"
    shared_keys = sorted(set(grader_maps[g_ref].keys()) & set(grader_maps[g_strict].keys()))
    rng = np.random.default_rng(20261123)
    perm = rng.permutation(len(shared_keys))
    ordered_keys = [shared_keys[i] for i in perm]

    y_all = np.array([phys_by_sha[k]["phys_maj"] for k in ordered_keys], dtype=float)
    f41_all = np.array([grader_maps[g_ref][k] for k in ordered_keys], dtype=float)
    f61_all = np.array([grader_maps[g_strict][k] for k in ordered_keys], dtype=float)

    # Scenario 1: Identical underlying replies (true delta = 0), but grader swaps GPT-4.1 -> GPT-6.1-sol
    # Scenario 2: Identical underlying replies (true delta = 0), prompt edit injects +6 pp false positives into GPT-4.1
    f41_hack = f41_all.copy()
    neg_idx = np.where((y_all == 0.0) & (f41_hack == 0.0))[0]
    hack_draws = rng.choice(neg_idx, size=min(len(neg_idx), int(round(0.06 * len(ordered_keys)))), replace=False)
    f41_hack[hack_draws] = 1.0

    ppi_audit = {}
    for m_lab in (50, 100, 200):
        # Split: last m_lab are clinician-labeled audit slice, rest unlabeled
        n_u = len(ordered_keys) - m_lab
        # Check shortcut scenario (f41 -> f41_hack, true delta = 0)
        res_hack = ppi_plus_plus_paired(
            f0_unlab=f41_all[:n_u], f1_unlab=f41_hack[:n_u],
            f0_lab=f41_all[n_u:], f1_lab=f41_hack[n_u:],
            y0_lab=y_all[n_u:], y1_lab=y_all[n_u:],
            alpha=0.05,
        )
        cal_hack = judge_calibration_equivalence(
            f0_lab=f41_all[n_u:], f1_lab=f41_hack[n_u:],
            y0_lab=y_all[n_u:], y1_lab=y_all[n_u:],
            tau_cal=0.02, alpha=0.05,
        )
        # Check grader-swap scenario (GPT-4.1 -> GPT-6.1-sol, true delta = 0)
        cal_swap = judge_calibration_equivalence(
            f0_lab=f41_all[n_u:], f1_lab=f61_all[n_u:],
            y0_lab=y_all[n_u:], y1_lab=y_all[n_u:],
            tau_cal=0.02, alpha=0.05,
        )
        ppi_audit[f"m_{m_lab}"] = {
            "shortcut_raw_judge_delta_pp": round(res_hack["raw_judge_delta"] * 100.0, 2),
            "shortcut_ppi_delta_pp": round(res_hack["ppi_delta"] * 100.0, 2),
            "shortcut_ppi_ci_95_pp": [round(res_hack["ci_lower"] * 100.0, 2), round(res_hack["ci_upper"] * 100.0, 2)],
            "shortcut_calibration_passed": cal_hack["calibration_passed"],
            "shortcut_alert_fired": cal_hack["shortcut_alert"],
            "grader_swap_41_to_61_diff_cal_pp": round(cal_swap["diff_cal_mean"] * 100.0, 2),
            "grader_swap_alert_fired": cal_swap["shortcut_alert"],
        }

    return {
        "n_shared_items": len(shared_keys),
        "graders_vs_physicians": summary_by_grader,
        "ppi_and_shortcut_audit": ppi_audit,
    }


def run_part_d_mvm_table() -> list[dict[str, Any]]:
    """Part D: Exact Minimum Verifiable Margin table across sample sizes and release cadences.

    Uses exact finite-sample Gauss-Chebyshev Jeffreys Beta(0.5, 0.5) mixture betting bounds
    and exact binomial-quantile power inversion (zero ad hoc scaling constants).
    """
    rows = []
    for n in (30, 50, 80, 100, 150, 200, 300, 500, 949, 2500, 10000, 25000):
        s1 = minimum_verifiable_margin(n=n, alpha=0.05, background_flip_rate=0.01, discordant_rate=0.05)
        q4 = minimum_verifiable_margin(n=n, alpha=0.05 / 4, background_flip_rate=0.01, discordant_rate=0.05)
        m12 = minimum_verifiable_margin(n=n, alpha=0.05 / 12, background_flip_rate=0.01, discordant_rate=0.05)
        rows.append({
            "n_cases": n,
            "single_release_zero_error_floor_pp": round(s1["zero_error_betting_floor"] * 100.0, 2),
            "single_release_exact_ceiling_1pct_bg_pp": round(s1["exact_betting_mvm"] * 100.0, 2),
            "single_release_empirical_mvm_pp": round(s1["exact_betting_margin_above_bg"] * 100.0, 2),
            "quarterly_4x_empirical_mvm_pp": round(q4["exact_betting_margin_above_bg"] * 100.0, 2),
            "monthly_12x_empirical_mvm_pp": round(m12["exact_betting_margin_above_bg"] * 100.0, 2),
            "single_release_fixed_horizon_z_mvm_pp": round(s1["fixed_horizon_mvm"] * 100.0, 2),
        })
    return rows


def main() -> None:
    results = {
        "meta": {
            "description": "Experiment 1 ($0 API cash): aggregate-only retrospective audit on saved clinical evaluations",
            "phi_included": False,
        },
        "part_a_compass_949_r3": run_part_a_compass_texts(),
        "part_b_hbpro_350_3graders": run_part_b_hbpro_graders(),
        "part_c_grader_meta_2000_6graders": run_part_c_grader_meta_ppi(),
        "part_d_mvm_power_table": run_part_d_mvm_table(),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote aggregate-only results to {OUT_JSON}")
    print(json.dumps(results, indent=2)[:4000])


if __name__ == "__main__":
    main()
