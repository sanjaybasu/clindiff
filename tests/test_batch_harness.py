"""Unit tests for the resumable Batch API prospective drift harness ($0 API calls)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts/03_run_prospective_drift.py"
PUB_VIG = Path(__file__).resolve().parents[1] / "vignettes/medicaid_safety_v0.1.jsonl"
CAN_VIG = Path(__file__).resolve().parents[1] / "vignettes/buyer_canary_template.jsonl"


def _load_mod():
    spec = importlib.util.spec_from_file_location("drift_harness", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_dry_run_stages_anthropic_and_openai_batches_with_unique_ids(tmp_path):
    mod = _load_mod()
    plan = mod.stage_run(
        vignette_paths=[PUB_VIG, CAN_VIG],
        models=["claude-sonnet-4-5", "gpt-5.4"],
        repeats=3,
        out_dir=tmp_path,
    )
    assert plan["api_calls_made"] == 0
    assert plan["n_vignettes"] == 350
    assert plan["total_est_batch_usd"] == round(plan["total_est_sync_usd"] * 0.5, 4)

    for m in plan["models"]:
        assert m["n_pending_requests"] == 1050
        lines = Path(m["batch_jsonl"]).read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1050
        cids = [json.loads(ln)["custom_id"] for ln in lines]
        assert len(set(cids)) == 1050


def test_checkpoint_resume_skips_completed_custom_ids(tmp_path):
    mod = _load_mod()
    ckpt = tmp_path / "gpt-4.1-2025-04-14_completed.jsonl"
    pub_rows = mod.load_vignettes([PUB_VIG])
    done_ids = [f"gpt-4.1-2025-04-14__{pub_rows[0]['vignette_id']}__r0", f"gpt-4.1-2025-04-14__{pub_rows[0]['vignette_id']}__r1"]
    ckpt.write_text("\n".join(json.dumps({"custom_id": c, "ok": True}) for c in done_ids) + "\n", encoding="utf-8")

    plan = mod.stage_run(
        vignette_paths=[PUB_VIG],
        models=["gpt-4.1-2025-04-14"],
        repeats=3,
        out_dir=tmp_path,
    )
    m = plan["models"][0]
    assert m["skipped_completed"] == 2
    assert m["n_pending_requests"] == 300 * 3 - 2
    pending_cids = {json.loads(ln)["custom_id"] for ln in Path(m["batch_jsonl"]).read_text(encoding="utf-8").splitlines()}
    assert pending_cids.isdisjoint(set(done_ids))
