"""CLI entry point for clindiff: evaluate, mvm, verify-bundle."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .gates import evaluate_release, verify_bundle_integrity
from .report import render_report_pdf
from .stats import minimum_verifiable_margin


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="clindiff", description="Self-hosted clinical AI change-control verifier.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("evaluate", help="Run Gates 0-3 on paired JSONL traces and emit signed JSON + PDF.")
    p_eval.add_argument("--traces", type=Path, required=True)
    p_eval.add_argument("--manifest", type=Path, required=True)
    p_eval.add_argument("--profile", default="enterprise_governance", choices=["enterprise_governance", "samd_pccp_optin"])
    p_eval.add_argument("--clinician-cal", type=Path, default=None)
    p_eval.add_argument("--out-dir", type=Path, required=True)

    p_mvm = sub.add_parser("mvm", help="Compute exact Minimum Verifiable Margin for a sample size N.")
    p_mvm.add_argument("--n", type=int, required=True)
    p_mvm.add_argument("--alpha", type=float, default=0.05)
    p_mvm.add_argument("--discordant-rate", type=float, default=0.05)
    p_mvm.add_argument("--releases-per-year", type=int, default=1)

    p_ver = sub.add_parser("verify-bundle", help="Verify cryptographic SHA-256 integrity of a change_gate_report.json.")
    p_ver.add_argument("bundle_json", type=Path)

    args = ap.parse_args(argv)

    if args.cmd == "evaluate":
        records = _read_jsonl(args.traces)
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        cal_records = _read_jsonl(args.clinician_cal) if args.clinician_cal else None
        report = evaluate_release(
            records=records,
            manifest=manifest,
            profile_name=args.profile,
            clinician_calibration_records=cal_records,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_json = args.out_dir / "change_gate_report.json"
        out_pdf = args.out_dir / "change_gate_report.pdf"
        out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        render_report_pdf(report, out_pdf)
        print(f"Decision: {report['decision']} | JSON: {out_json} | PDF: {out_pdf}")
        return 0

    if args.cmd == "mvm":
        adj_alpha = args.alpha / max(1, args.releases_per_year)
        res = minimum_verifiable_margin(n=args.n, alpha=adj_alpha, discordant_rate=args.discordant_rate)
        print(json.dumps(res, indent=2))
        return 0

    if args.cmd == "verify-bundle":
        report = json.loads(args.bundle_json.read_text(encoding="utf-8"))
        ok = verify_bundle_integrity(report)
        if ok:
            print(f"OK: bundle_sha256 verified ({report['bundle_sha256']})")
            return 0
        print("FAILED: bundle_sha256 mismatch (file tampered or unsigned)", file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
