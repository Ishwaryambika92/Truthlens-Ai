"""Generate a branded PDF report for a single Analysis using ReportLab."""
import io

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)


def build_analysis_pdf(analysis, fact_checks) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=22 * mm, bottomMargin=18 * mm, leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TLTitle", parent=styles["Title"], textColor=colors.HexColor("#0b1220"))
    h2 = ParagraphStyle("TLH2", parent=styles["Heading2"], textColor=colors.HexColor("#1c8a99"), spaceBefore=14)
    body = ParagraphStyle("TLBody", parent=styles["BodyText"], leading=15)
    muted = ParagraphStyle("TLMuted", parent=styles["BodyText"], textColor=colors.grey, fontSize=9)

    story = []
    story.append(Paragraph("TruthLens AI — Verification Report", title_style))
    story.append(Paragraph(f"Generated {analysis.created_at.strftime('%B %d, %Y at %H:%M UTC')}", muted))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#dddddd"), spaceBefore=8, spaceAfter=14))

    from detector.verification.canonical import build_canonical_verification_result
    canonical = build_canonical_verification_result(analysis=analysis, fact_checks=fact_checks)
    final_verdict = canonical["final_assessment"]["verdict"]

    story.append(Paragraph("Final Assessment", h2))
    era_label = "Current / Present News" if getattr(analysis, "temporal_context", "") == "current" else (
        "Historical / Past News" if getattr(analysis, "temporal_context", "") == "historical" else "Unspecified Era"
    )
    final_data = [
        ["Final Assessment", final_verdict],
        ["Verification Confidence", f"{canonical['final_assessment']['confidence_label']}"
                                     f" — {canonical['final_assessment']['confidence_score']:.0f}%"],
        ["Temporal Context / Era", era_label],
    ]
    ft = Table(final_data, colWidths=[160, 320])
    ft.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
    ]))
    story.append(ft)
    for line in (analysis.verification_reasoning or []):
        story.append(Paragraph(f"• {line}", body))

    story.append(Paragraph("ML Model", h2))
    pred_display_map = {"REAL": "Likely Real", "FAKE": "Likely Fake", "UNCERTAIN": "Uncertain"}
    pred_text = pred_display_map.get(analysis.prediction, "Unavailable")
    conf_text = f"{analysis.confidence}%" if analysis.confidence is not None else "N/A"
    conf_status = f" ({analysis.confidence_status} confidence)" if analysis.confidence_status else ""
    data = [["Model Prediction", pred_text], ["Model Confidence", conf_text + conf_status], ["Input Type", analysis.get_input_type_display()]]
    if analysis.source_url:
        data.append(["Source URL", analysis.source_url])
    t = Table(data, colWidths=[120, 360])
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#eeeeee")),
    ]))
    story.append(t)

    story.append(Paragraph("Explanation", h2))
    if analysis.explanation:
        for line in analysis.explanation:
            story.append(Paragraph(line, body))
    else:
        story.append(Paragraph("No explanation available.", body))

    story.append(Paragraph("Clickbait Analysis", h2))
    story.append(Paragraph(
        f"Score: {analysis.clickbait_score}% — Classification: {analysis.clickbait_classification}", body
    ))
    for s in analysis.clickbait_signals:
        story.append(Paragraph(f"• {s}", body))

    story.append(Paragraph("AI-Generated Text Analysis (Experimental)", h2))
    story.append(Paragraph(f"Result: {analysis.ai_text_label}", body))
    story.append(Paragraph(
        "AI-generated text detection is experimental and may produce incorrect results.", muted
    ))

    story.append(Paragraph("Source Analysis", h2))
    if analysis.source_analysis and analysis.source_analysis.get("available"):
        story.append(Paragraph(f"Domain: {analysis.source_analysis.get('domain', 'N/A')}", body))
        for sig in analysis.source_analysis.get("signals", []):
            story.append(Paragraph(f"• {sig}", body))
    else:
        story.append(Paragraph("No URL was supplied, so no source/domain analysis is available.", body))

    story.append(Paragraph("Fact Verification", h2))
    if fact_checks:
        fc_data = [["Claim", "Status", "Source"]]
        for fc in fact_checks:
            fc_data.append([fc.claim[:70], fc.status, fc.source])
        fc_table = Table(fc_data, colWidths=[260, 100, 120])
        fc_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
        ]))
        story.append(fc_table)
    else:
        story.append(Paragraph(
            "Fact verification unavailable or no matching fact-checks were found.", body
        ))

    from detector.verification.claims import build_authoritative_claims_structure
    claims_struct = build_authoritative_claims_structure(
        claims_list=analysis.claims,
        evidence_data=analysis.evidence,
        fact_checks=fact_checks,
    )
    if claims_struct["claims"]:
        story.append(Paragraph("Claim Verification Summary", h2))
        cc = claims_struct["claim_counts"]
        sc = claims_struct["source_counts"]
        story.append(Paragraph(
            f"<b>Claims Analyzed:</b> {cc['total']} &nbsp;|&nbsp; "
            f"<b>Supporting Claims:</b> {cc['supported']} &nbsp;|&nbsp; "
            f"<b>Contradicting Claims:</b> {cc['contradicted']} &nbsp;|&nbsp; "
            f"<b>Insufficient:</b> {cc['insufficient']} &nbsp;|&nbsp; "
            f"<b>Sources Evaluated:</b> {sc['total']}",
            body
        ))
        claims_table_data = [["#", "Claim", "Status"]]
        for c in claims_struct["claims"][:6]:
            claims_table_data.append([str(c["index"]), c["claim"][:75], c["status"]])
        ct = Table(claims_table_data, colWidths=[25, 355, 100])
        ct.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f2f2")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
        ]))
        story.append(ct)

    story.append(Paragraph("Summary", h2))
    story.append(Paragraph(analysis.summary or "No summary available.", body))

    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", color=colors.HexColor("#dddddd")))
    story.append(Paragraph(
        "TruthLens AI — automated analysis. This report supports human judgment; it is not a "
        "definitive fact-checking authority.", muted
    ))

    doc.build(story)
    return buffer.getvalue()
