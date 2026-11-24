"""Pure-black Arial 1-page PDF executive governance report generator (zero color)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _register_fonts() -> tuple[str, str, str]:
    reg = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    ital = Path("/System/Library/Fonts/Supplemental/Arial Italic.ttf")
    if reg.exists() and bold.exists():
        if "ClinDiffArial" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("ClinDiffArial", str(reg)))
            pdfmetrics.registerFont(TTFont("ClinDiffArial-Bold", str(bold)))
            if ital.exists():
                pdfmetrics.registerFont(TTFont("ClinDiffArial-Italic", str(ital)))
                pdfmetrics.registerFontFamily(
                    "ClinDiffArial",
                    normal="ClinDiffArial",
                    bold="ClinDiffArial-Bold",
                    italic="ClinDiffArial-Italic",
                    boldItalic="ClinDiffArial-Bold",
                )
            else:
                pdfmetrics.registerFontFamily(
                    "ClinDiffArial",
                    normal="ClinDiffArial",
                    bold="ClinDiffArial-Bold",
                    italic="ClinDiffArial",
                    boldItalic="ClinDiffArial-Bold",
                )
        return ("ClinDiffArial", "ClinDiffArial-Bold", "ClinDiffArial-Italic" if ital.exists() else "ClinDiffArial")
    return ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique")


def render_report_pdf(report: dict[str, Any], out_pdf: Path) -> None:
    """Render a strictly 1-page black-and-white executive report from a signed gate report."""
    f_reg, f_bold, f_ital = _register_fonts()
    black = colors.black
    gray = colors.HexColor("#666666")
    shade = colors.HexColor("#F2F2F2")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=0.60 * inch,
        rightMargin=0.60 * inch,
        topMargin=0.50 * inch,
        bottomMargin=0.45 * inch,
    )
    w = letter[0] - 1.20 * inch

    s_title = ParagraphStyle("t", fontName=f_bold, fontSize=11.5, leading=14, textColor=black)
    s_sub = ParagraphStyle("sub", fontName=f_ital, fontSize=8.4, leading=10.8, textColor=black, spaceAfter=4)
    s_sec = ParagraphStyle("sec", fontName=f_bold, fontSize=9.4, leading=11.8, textColor=black, spaceBefore=7, spaceAfter=3)
    s_body = ParagraphStyle("b", fontName=f_reg, fontSize=8.7, leading=11.2, textColor=black, spaceAfter=4)
    s_cell = ParagraphStyle("c", fontName=f_reg, fontSize=8.0, leading=10.2, textColor=black)
    s_cell_b = ParagraphStyle("cb", fontName=f_bold, fontSize=8.0, leading=10.2, textColor=black)
    s_foot = ParagraphStyle("f", fontName=f_reg, fontSize=7.3, leading=9.5, textColor=black)

    m = report["manifest"]
    g0 = report["gate0"]
    g1 = report["gate1"]
    ov = g1["overall"]
    q = g1["quality"]
    ch = g1["churn"]
    mvm = g1["minimum_verifiable_margin"]

    story = []
    sys_name = m.get("system_name", "Clinical AI System")
    cand = m.get("candidate_version", "candidate")
    prior = m.get("prior_version", "prior")
    story.append(Paragraph(f"Pre-Release AI Change-Control Report: {sys_name} ({cand})", s_title))
    story.append(Paragraph(f"Profile: {report['profile']} | Decision: {report['decision']} | Generated: {report['generated_utc']}", s_sub))

    rule = Table([[""]], colWidths=[w], rowHeights=[1])
    rule.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 1.0, black), ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]))
    story.append(rule)
    story.append(Spacer(1, 5))

    # Section 1: Configuration & Decision
    story.append(Paragraph("1. Configuration and release decision", s_sec))
    meta = [
        [Paragraph("<b>System / release</b>", s_cell), Paragraph(f"{sys_name} {cand} vs. {prior}", s_cell),
         Paragraph("<b>Decision</b>", s_cell), Paragraph(f"<b>{report['decision']}</b>", s_cell)],
        [Paragraph("<b>Pinned model ID</b>", s_cell), Paragraph(str(m.get("model_id", "unspecified")), s_cell),
         Paragraph("<b>Manifest SHA-256</b>", s_cell), Paragraph(report["manifest_sha256"][:24] + "...", s_cell)],
        [Paragraph("<b>Declared scope</b>", s_cell), Paragraph(str(m.get("declared_scope", "See statement of work")), s_cell),
         Paragraph("<b>Smallest detectable change</b>", s_cell), Paragraph(f"±{100.0 * mvm['empirical_mvm']:.1f} pp (N={mvm['n']}, R={ov['r1']}, 95% conf.)", s_cell)],
    ]
    mt = Table(meta, colWidths=[w * 0.20, w * 0.32, w * 0.22, w * 0.26])
    mt.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, black),
        ("LINEBELOW", (0, 0), (-1, -2), 0.35, gray),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(mt)

    # Section 2: Paired Verification Summary
    story.append(Paragraph("2. Paired test results", s_sec))
    th = report["thresholds"]
    rows = [
        [Paragraph("Check", s_cell_b), Paragraph("Agreed rule", s_cell_b), Paragraph("Observed result", s_cell_b), Paragraph("Status", s_cell_b)],
        [
            Paragraph("Hard safety rules (Gate 0)", s_cell),
            Paragraph(f"≤ {g0['max_allowed']} failures", s_cell),
            Paragraph(f"{g0['failures_v1']} of {g0['n_cases']} failed (prior: {g0['failures_v0']})", s_cell),
            Paragraph("PASS" if g0["passed"] else "HOLD", s_cell_b),
        ],
        [
            Paragraph("Average quality change vs. prior", s_cell),
            Paragraph(f"95% CI > {100.0 * th['net_quality_ni_margin']:.1f} pp", s_cell),
            Paragraph(f"{100.0 * q['net_delta']:+.1f} pp (95% CI, {100.0 * q['ci_lower']:+.1f} to {100.0 * q['ci_upper']:+.1f})", s_cell),
            Paragraph("PASS" if q["passed"] else "HOLD", s_cell_b),
        ],
        [
            Paragraph("Safe-to-unsafe flip rate (upper 95% bound)" if not g1.get("stratum_split") else "Safety-positive missed-hazard flip (95% UB)", s_cell),
            Paragraph(f"≤ {100.0 * th['max_gross_flip_upper']:.1f}%", s_cell),
            Paragraph(
                f"Mean {100.0 * g1['stratum_split']['safety_positive']['psi_down_mean']:.1f}%, UB {100.0 * g1['stratum_split']['safety_positive']['psi_down_upper']:.1f}% (n={g1['stratum_split']['safety_positive']['n_cases']})"
                if g1.get("stratum_split")
                else f"Mean {100.0 * ov['psi_down_mean']:.1f}%, upper bound {100.0 * ov['psi_down_upper']:.1f}%",
                s_cell,
            ),
            Paragraph("PASS" if g1["checks"]["flip_bound_ok"] else "HOLD", s_cell_b),
        ],
    ]
    if g1.get("stratum_split") is not None:
        rt_sp = g1["stratum_split"]["routine"]
        rows.append([
            Paragraph("Routine false-escalation shift", s_cell),
            Paragraph(f"≤ +{100.0 * rt_sp['max_allowed_delta']:.1f} pp", s_cell),
            Paragraph(f"{100.0 * rt_sp['false_escalation_delta']:+.1f} pp (spec. {100.0 * rt_sp['specificity_v0']:.1f}% → {100.0 * rt_sp['specificity_v1']:.1f}%, n={rt_sp['n_cases']})", s_cell),
            Paragraph("PASS" if g1["checks"].get("routine_budget_ok", True) else "HOLD", s_cell_b),
        ])
    rows.append(
        [
            Paragraph("Unbiased case-level churn", s_cell),
            Paragraph(f"≤ {th['max_churn_upper']:.3f}", s_cell),
            Paragraph(f"{ch['theta_sq_mean']:.3f} (upper bound {ch['theta_sq_upper']:.3f})" if ch else "Requires R ≥ 2", s_cell),
            Paragraph("PASS" if g1["checks"]["churn_ok"] else "HOLD", s_cell_b),
        ]
    )
    if g1["anchor"] is not None:
        anc = g1["anchor"]
        rows.append([
            Paragraph("Cumulative drift vs. initial anchor", s_cell),
            Paragraph(f"≤ +{100.0 * th['anchor_max_drift']:.1f} pp harm", s_cell),
            Paragraph(f"{100.0 * anc['net_harm_vs_anchor']:+.1f} pp (flip upper {100.0 * anc['psi_down_vs_anchor_upper']:.1f}%)", s_cell),
            Paragraph("PASS" if anc["passed"] else "HOLD", s_cell_b),
        ])
    if report["gate2"]["run"]:
        cal = report["gate2"]["calibration"]
        rows.append([
            Paragraph("Blinded clinician judge calibration (Gate 2)", s_cell),
            Paragraph(f"|bias| < {100.0 * cal['tau_cal']:.1f} pp", s_cell),
            Paragraph(f"{100.0 * cal['diff_cal_mean']:+.1f} pp (m={cal['m_labeled']})", s_cell),
            Paragraph("PASS" if report["gate2"]["passed"] else "HOLD", s_cell_b),
        ])

    rt = Table(rows, colWidths=[w * 0.34, w * 0.19, w * 0.37, w * 0.10])
    rt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), shade),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, black),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, gray),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    story.append(rt)

    # Section 3: Subgroup Breakdown
    story.append(Paragraph("3. Subgroup breakdown", s_sec))
    sub_rows = [[
        Paragraph("Stratum", s_cell_b),
        Paragraph("N", s_cell_b),
        Paragraph("Prior safety", s_cell_b),
        Paragraph("New safety", s_cell_b),
        Paragraph("Flip mean (95% UB)", s_cell_b),
        Paragraph("Churn", s_cell_b),
    ]]
    for st, vals in list(g1["subgroups"].items())[:6]:
        sub_rows.append([
            Paragraph(st, s_cell),
            Paragraph(str(vals["n_cases"]), s_cell),
            Paragraph(f"{100.0 * vals['safety_v0_mean']:.1f}%", s_cell),
            Paragraph(f"{100.0 * vals['safety_v1_mean']:.1f}%", s_cell),
            Paragraph(f"{100.0 * vals['psi_down_mean']:.1f}% ({100.0 * vals['psi_down_upper']:.1f}%)", s_cell),
            Paragraph(f"{vals['theta_sq_mean']:.3f}" if vals["theta_sq_mean"] is not None else "n/a", s_cell),
        ])
    st_tbl = Table(sub_rows, colWidths=[w * 0.24, w * 0.08, w * 0.16, w * 0.16, w * 0.24, w * 0.12])
    st_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), shade),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, black),
        ("LINEBELOW", (0, 1), (-1, -2), 0.35, gray),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3.0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.0),
    ]))
    story.append(st_tbl)

    story.append(Spacer(1, 6))
    story.append(Paragraph(f"Bundle SHA-256: {report['bundle_sha256']}", s_body))
    if report["blocking_reasons"]:
        story.append(Paragraph("Blocking reasons: " + " | ".join(report["blocking_reasons"]), s_body))

    ft = Table([[Paragraph(report["regulatory_notice"], s_foot)]], colWidths=[w])
    ft.setStyle(TableStyle([("LINEABOVE", (0, 0), (-1, 0), 0.6, black), ("TOPPADDING", (0, 0), (-1, -1), 4), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story.append(ft)

    doc.build(story)
