"""
Email and PDF report template for Cardiac Risk MVP.

This file contains:
- build_report_pdf(prediction, hero_image_path=None)
- send_report_email(recipient, pdf_bytes)

Secrets expected from Streamlit:
GMAIL_SENDER
GMAIL_APP_PASSWORD
"""

from __future__ import annotations

import io
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DEFAULT_SENDER = "khan.abdullah135790@gmail.com"


def _secret(name: str, default: str | None = None) -> str | None:
    """Read a value from Streamlit Secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


def build_report_pdf(prediction: dict, hero_image_path: str | Path | None = None) -> bytes:
    """Generate the Cardiac Risk MVP PDF report."""
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Cardiac Risk Assessment",
        author="Cardiac Risk MVP",
    )

    styles = getSampleStyleSheet()

    title = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#063B78"),
        alignment=TA_CENTER,
        spaceAfter=5,
    )

    subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    section = ParagraphStyle(
        "ReportSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#063B78"),
        spaceBefore=8,
        spaceAfter=8,
    )

    body = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#172033"),
        spaceAfter=5,
    )

    small = ParagraphStyle(
        "ReportSmall",
        parent=body,
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # Optional MVP hero artwork.
    if hero_image_path:
        image_path = Path(hero_image_path)
        if image_path.exists():
            try:
                img = RLImage(
                    str(image_path),
                    width=178 * mm,
                    height=48 * mm,
                )
                img.hAlign = "CENTER"
                story.extend([img, Spacer(1, 5 * mm)])
            except Exception:
                pass

    story.append(Paragraph("CARDIAC RISK ASSESSMENT", title))
    story.append(
        Paragraph(
            f"Generated {prediction.get('timestamp', 'N/A')} · Cardiac Risk MVP",
            subtitle,
        )
    )

    metric_data = [
        [
            Paragraph("<b>Framingham Score</b>", body),
            Paragraph("<b>Est. Yrs to Event</b>", body),
            Paragraph("<b>10-Yr CVD Risk</b>", body),
            Paragraph("<b>Risk Category</b>", body),
        ],
        [
            Paragraph(f"{float(prediction.get('framingham_score', 0)):.1f} pts", title),
            Paragraph(f"{float(prediction.get('years_to_event', 0)):.1f} yrs", title),
            Paragraph(f"{float(prediction.get('cvd_risk', 0)):.1f}%", title),
            Paragraph(str(prediction.get("risk_category", "N/A")), title),
        ],
    ]

    metric_table = Table(
        metric_data,
        colWidths=[43 * mm] * 4,
        rowHeights=[12 * mm, 20 * mm],
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F4F8FC")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#B8CADC")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D7E2EC")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(metric_table)

    story.append(Spacer(1, 7 * mm))
    story.append(Paragraph("10-Year CVD Risk Profile", section))

    risk = max(0.0, min(100.0, float(prediction.get("cvd_risk", 0))))
    risk_table = Table(
        [[Paragraph(f"<b>{risk:.1f}% estimated risk</b>", body)]],
        colWidths=[172 * mm],
        rowHeights=[13 * mm],
    )
    risk_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EAF3FB")),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#9FB9D1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    story.append(risk_table)

    story.append(Paragraph("Possible Suggestions", section))

    suggestions = prediction.get("suggestions") or [
        "Discuss the model result with a qualified healthcare professional.",
        "Continue monitoring established cardiovascular risk factors.",
        "Maintain healthy lifestyle habits appropriate to your circumstances.",
    ]

    for suggestion in suggestions:
        story.append(Paragraph(f"• {suggestion}", body))

    story.append(Paragraph("Model Inputs", section))

    input_rows = []
    for key, value in prediction.get("inputs", {}).items():
        if pd.isna(value):
            value = "N/A"

        input_rows.append(
            [
                Paragraph(str(key), body),
                Paragraph(str(value), body),
            ]
        )

    if input_rows:
        input_table = Table(
            input_rows,
            colWidths=[90 * mm, 82 * mm],
        )
        input_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F5F8FB")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD7E3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(input_table)

    story.append(Spacer(1, 7 * mm))
    story.append(Paragraph("Important Disclaimer", section))
    story.append(
        Paragraph(
            "This report is generated by a machine-learning MVP for research and "
            "educational purposes. The estimates are not a medical diagnosis, "
            "do not establish that a cardiac event will occur, and should not "
            "replace assessment by a qualified healthcare professional. Seek "
            "appropriate medical care for concerning or urgent symptoms.",
            small,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def send_report_email(
    recipient: str,
    pdf_bytes: bytes,
    sender: str | None = None,
) -> None:
    """Send the generated PDF through Gmail SMTP."""
    sender = sender or _secret("GMAIL_SENDER", DEFAULT_SENDER)
    app_password = _secret("GMAIL_APP_PASSWORD")

    if not sender:
        raise RuntimeError("GMAIL_SENDER is not configured.")

    if not app_password:
        raise RuntimeError(
            "GMAIL_APP_PASSWORD is not configured. Add a Google App Password "
            "to .streamlit/secrets.toml."
        )

    msg = EmailMessage()
    msg["Subject"] = "Cardiac Risk Assessment Report"
    msg["From"] = sender
    msg["To"] = recipient
    msg.set_content(
        "Dear recipient,\n\n"
        "Please find attached your Cardiac Risk Assessment report generated "
        "by the Cardiac Risk MVP.\n\n"
        "This report is intended for research/educational use and is not a "
        "medical diagnosis or a substitute for assessment by a qualified "
        "healthcare professional.\n\n"
        "Regards,\n"
        "Cardiac Risk MVP"
    )

    msg.add_attachment(
        pdf_bytes,
        maintype="application",
        subtype="pdf",
        filename="cardiac_risk_assessment.pdf",
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(msg)