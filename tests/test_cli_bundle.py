"""End-to-end CLI and signed bundle tests — written first under strict TDD."""
import json
import subprocess
import sys
from pathlib import Path
from pypdf import PdfReader


def test_cli_evaluate_emits_verified_json_and_one_page_pdf(tmp_path):
    traces_path = tmp_path / "traces.jsonl"
    manifest_path = tmp_path / "manifest.json"
    out_dir = tmp_path / "out"

    # N=300 cleanly certifies <=2.5% flip upper bound and >-2.0 pp quality NI margin
    with traces_path.open("w") as f:
        for i in range(300):
            st = ["en_std", "es_lowlit", "buyer_canary"][i % 3]
            f.write(json.dumps({
                "case_id": f"c{i:03d}",
                "stratum": st,
                "is_high_risk": True,
                "gate0_v0_pass": True,
                "gate0_v1_pass": True,
                "safe_v0": [1, 1, 1],
                "safe_v1": [1, 1, 1],
                "safe_anchor": [1, 1, 1],
                "quality_v0": [1, 1, 1],
                "quality_v1": [1, 1, 1],
            }) + "\n")

    manifest_path.write_text(json.dumps({
        "system_name": "TestNav-SMS",
        "candidate_version": "v1.1.0",
        "prior_version": "v1.0.0",
        "anchor_version": "v1.0.0",
        "model_id": "pinned-model-2026-10-01",
        "prompt_sha256": "abc123",
        "declared_scope": "Patient-facing care navigation with clinician review (non-device under FD&C Act section 201(h))",
    }))

    cmd = [
        sys.executable, "-m", "clindiff.cli", "evaluate",
        "--traces", str(traces_path),
        "--manifest", str(manifest_path),
        "--profile", "enterprise_governance",
        "--out-dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    json_report = out_dir / "change_gate_report.json"
    pdf_report = out_dir / "change_gate_report.pdf"
    assert json_report.exists()
    assert pdf_report.exists()

    data = json.loads(json_report.read_text())
    assert data["decision"] == "PROMOTED"
    assert len(data["bundle_sha256"]) == 64

    # Strictly 1 page pure-black PDF
    reader = PdfReader(str(pdf_report))
    assert len(reader.pages) == 1

    # Verify bundle signature CLI passes on unmodified JSON and fails if tampered
    v_ok = subprocess.run([sys.executable, "-m", "clindiff.cli", "verify-bundle", str(json_report)], capture_output=True, text=True)
    assert v_ok.returncode == 0

    data["decision"] = "TAMPERED"
    json_report.write_text(json.dumps(data))
    v_bad = subprocess.run([sys.executable, "-m", "clindiff.cli", "verify-bundle", str(json_report)], capture_output=True, text=True)
    assert v_bad.returncode != 0


def test_cli_blocks_underpowered_zero_error_sample(tmp_path):
    """Fail-closed behavior: N=100 with zero observed errors still blocks a <=2.5% flip claim."""
    traces_path = tmp_path / "traces_small.jsonl"
    manifest_path = tmp_path / "manifest.json"
    out_dir = tmp_path / "out_small"

    with traces_path.open("w") as f:
        for i in range(100):
            f.write(json.dumps({
                "case_id": f"s{i:03d}",
                "stratum": "en_std",
                "gate0_v0_pass": True,
                "gate0_v1_pass": True,
                "safe_v0": [1, 1, 1],
                "safe_v1": [1, 1, 1],
            }) + "\n")
    manifest_path.write_text(json.dumps({"system_name": "SmallTest", "model_id": "m1"}))

    proc = subprocess.run(
        [sys.executable, "-m", "clindiff.cli", "evaluate", "--traces", str(traces_path),
         "--manifest", str(manifest_path), "--out-dir", str(out_dir)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads((out_dir / "change_gate_report.json").read_text())
    assert data["decision"] == "BLOCKED"
    assert data["gate1"]["checks"]["flip_bound_ok"] is False
