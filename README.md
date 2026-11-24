# `clindiff`: Self-Hosted Statistical Change-Control Verification for Clinical AI Updates

`clindiff` is an Apache-2.0, self-hosted Python library, CLI, and zero-PHI multilingual Medicaid benchmark for verifying clinical artificial intelligence model upgrades, fine-tuning cycles, prompt edits, API parameter defaults, and automated-grader substitutions inside a deploying organization's own infrastructure. Accompanies our manuscript *"Statistical Change-Control Verification for Clinical Artificial Intelligence Updates"* (Basu S, Sheth P, Batniji R; under review, 2026). Interactive Minimum Verifiable Margin (MVM) calculator and procurement specification: [https://sanjaybasu.github.io/clindiff/](https://sanjaybasu.github.io/clindiff/).

## What this repository provides

- **`src/clindiff/stats.py`:** Exact finite-sample anytime-valid betting e-processes and confidence sequences using a fixed 32-point Gauss-Chebyshev quadrature mixture of the Jeffreys arcsine prior $\mathrm{Beta}(0.5, 0.5)$ (valid by Ville's inequality and order-invariant); gross safe-to-unsafe downward flip rate ($\psi_{\mathrm{down}}$) and baseline repeat self-instability ($\psi_{\mathrm{self}}$); unbiased mean-squared case churn ($\hat{\theta}^2$); cluster-exact aggregation (`cluster_id`) for matched multilingual vignettes; exact binomial-quantile Minimum Verifiable Margin ($\Delta_{\min}$, zero heuristic multipliers); clinical-domain `profile_product_upper_bound`; online `e-LOND` sequential false discovery rate control; and split-calibrated paired PPI++ with fail-closed judge calibration equivalence testing.
- **`src/clindiff/gates.py`:** Three-gate release orchestration (Gate 0 deterministic invariants, Gate 1 paired safety/churn/subgroup/anchor/power-feasibility checks with stratum-separated safety-positive vs. routine alert budgets, and Gate 2 clinician-calibrated judge audit), live `evaluate_release_sequence()` with online `e-LOND`, and SHA-256 manifest and bundle signing.
- **`src/clindiff/guardrails.py`:** Zero-leakage deterministic Gate 0 clinical safety-floor post-processor operating strictly on raw patient input text and raw model output text (enforcing localized English/Spanish 988/911 crisis routing and autonomous dosing refusal).
- **`src/clindiff/report.py` & `cli.py`:** Command-line interface emitting signed JSON bundles and strictly one-page pure-black Arial governance PDFs.
- **`profiles/`:** Vendor-neutral governance profiles (`enterprise_governance.yaml` mapped to HHS Section 1557 [45 CFR 92.210], ONC HTI-1 voluntary templates post-HTI-5, CHAI Applied Model Cards, Joint Commission, and NCQA oversight; `multilingual_screening.yaml` calibrated for matched 100-scenario multilingual suites; and isolated `samd_pccp_optin.yaml`).
- **`vignettes/`:** Zero-PHI synthetic Medicaid benchmark (`medicaid_safety_v0.1.jsonl`, 300 vignettes = 100 independent clinical scenarios paired across standard English, low-literacy English paraphrase, and Spanish; 240 safety-positive + 60 routine navigation) and 50 disjoint buyer-canary templates (`buyer_canary_template.jsonl`).
- **`docs/index.html`:** Static, zero-dependency interactive Minimum Verifiable Margin calculator and vendor procurement clause template.

## Reproducibility and data

**No patient data, protected health information, or study result files are included in this repository.** All clinical text evaluations in the companion study were conducted within the Waymark network (the largest integrated Medicaid population health provider nationally) under WCG IRB review with a waiver of informed consent, and member-level data are protected under Waymark's data-governance policy and are not shared. Only code, governance profiles, unit/Monte Carlo validation tests, and the synthetic zero-PHI benchmark generator and templates reside here. Reasonable requests for collaborative aggregate re-analysis can be directed to the corresponding author (`sanjay.basu@ucsf.edu`).

## Layout

```text
.
├── docs/                  # Static GitHub Pages interactive MVM calculator & procurement clause
├── profiles/              # YAML governance profiles (enterprise, multilingual_screening, opt-in PCCP)
├── scripts/               # Staged reproducible evaluation, scoring, and dry-run batch scripts
├── src/clindiff/          # Core statistical primitives, release gates, guardrails, PDF report, CLI
├── tests/                 # Unit, Monte Carlo Type I error, cluster-exact, CLI, and zero-PHI tests
├── vignettes/             # Synthetic zero-PHI Medicaid benchmark (300 paired) & 50 canary templates
├── pyproject.toml         # Package configuration
└── LICENSE                # Apache License 2.0
```

## Quickstart

```bash
pip install -e .
pytest
```

Compute exact finite-sample Minimum Verifiable Margins ($\Delta_{\min}$):

```bash
clindiff mvm --n 300 --alpha 0.05 --discordant-rate 0.05 --releases-per-year 4
```

Evaluate paired multi-run traces locally (zero external transmission):

```bash
clindiff evaluate \
  --traces path/to/paired_traces.jsonl \
  --manifest path/to/release_manifest.json \
  --profile enterprise_governance \
  --out-dir out/release_check

clindiff verify-bundle out/release_check/change_gate_report.json
```

Regenerate the synthetic zero-PHI vignettes deterministically or stage zero-cost dry-run batch manifests (no network calls by default):

```bash
python3 scripts/02_build_synthetic_vignettes.py
python3 scripts/03_run_prospective_drift.py --include-canary
```

The full local test suite (`pytest`, 29 tests including Monte Carlo anytime-valid Type I error verification and cluster-exact invariance checks) runs in ~7 seconds on CPU with no external data or API keys required.

## Study context

Conducted within the Waymark network (the largest integrated Medicaid population health provider nationally), providing free medical and social services to patients receiving Medicaid, in collaboration with the University of California San Francisco Department of Medicine and the University of Pennsylvania School of Social Policy and Practice.

## License

Apache License 2.0 (see `LICENSE`).
