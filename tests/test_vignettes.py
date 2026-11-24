"""Validation tests for the zero-PHI synthetic Medicaid vignette suite and buyer canary template."""
import json
import re
from collections import Counter
from pathlib import Path

VIG_DIR = Path(__file__).resolve().parents[1] / "vignettes"
PUB_PATH = VIG_DIR / "medicaid_safety_v0.1.jsonl"
CAN_PATH = VIG_DIR / "buyer_canary_template.jsonl"

_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def _load(p: Path):
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_public_suite_counts_and_exact_three_way_pairing():
    rows = _load(PUB_PATH)
    assert len(rows) == 300
    by_base = Counter(r["base_id"] for r in rows)
    assert len(by_base) == 100
    assert all(v == 3 for v in by_base.values())
    strata = Counter(r["stratum"] for r in rows)
    assert strata == {"en_standard": 100, "en_lowlit_paraphrase": 100, "es_spanish": 100}
    domains = Counter(r["domain"] for r in rows)
    assert len(domains) == 5 and all(v == 60 for v in domains.values())
    sp = Counter(r["is_safety_positive"] for r in rows)
    assert sp == {True: 240, False: 60}


def test_buyer_canary_template_counts_and_zero_overlap():
    pub = _load(PUB_PATH)
    can = _load(CAN_PATH)
    assert len(can) == 50
    domains = Counter(r["domain"] for r in can)
    assert len(domains) == 5 and all(v == 10 for v in domains.values())
    assert set(r["vignette_id"] for r in pub).isdisjoint(set(r["vignette_id"] for r in can))
    assert set(r["prompt"].strip().lower() for r in pub).isdisjoint(set(r["prompt"].strip().lower() for r in can))


def test_zero_phi_patterns_in_vignettes_and_aggregate_results():
    for path in [PUB_PATH, CAN_PATH, VIG_DIR.parent / "results/retrospective_audit_aggregate.json"]:
        if not path.exists():
            continue
        txt = path.read_text(encoding="utf-8")
        assert _PHONE_RE.search(txt) is None, f"Phone pattern in {path}"
        assert _SSN_RE.search(txt) is None, f"SSN pattern in {path}"
        assert _EMAIL_RE.search(txt) is None, f"Email pattern in {path}"
        assert "F0001" not in txt and "waymarkPatientNumber" not in txt
