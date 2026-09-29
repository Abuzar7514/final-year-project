from datetime import datetime
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


# ---------------------------------------------------------
# Report theme
# ---------------------------------------------------------

HOSPITAL_BLUE = colors.HexColor("#0B5D8C")
LIGHT_BLUE = colors.HexColor("#EAF5FB")
VERY_LIGHT_BLUE = colors.HexColor("#F5FAFD")
DARK_TEXT = colors.HexColor("#1F2937")
MUTED_TEXT = colors.HexColor("#5B6777")
LIGHT_BORDER = colors.HexColor("#C8DCE8")
SUCCESS_BACKGROUND = colors.HexColor("#E8F6EF")
SUCCESS_BORDER = colors.HexColor("#2E8B57")
WARNING_BACKGROUND = colors.HexColor("#FFF8E8")
WARNING_BORDER = colors.HexColor("#D69E2E")


def safe_text(value, fallback="Not provided"):
    """
    Convert a value into safe text for a ReportLab Paragraph.
    """
    if value is None:
        return fallback

    value = str(value).strip()

    if not value:
        return fallback

    return escape(value)


def format_confidence(value):
    """
    Convert confidence into a clean percentage.

    Supports:
    - 0.93
    - 93
    - "93.42"
    - "93.42%"
    """
    if value is None:
        return "Not available"

    try:
        cleaned = str(value).replace("%", "").strip()
        number = float(cleaned)

        if 0 <= number <= 1:
            number *= 100

        return f"{number:.2f}%"

    except (TypeError, ValueError):
        return safe_text(value, "Not available")


def format_temperature(value):
    if value is None or str(value).strip() == "":
        return "Not provided"

    text = str(value).strip()

    if "°" in text or "f" in text.lower() or "c" in text.lower():
        return safe_text(text)

    return safe_text(f"{text} °F")


def format_duration(value):
    if value is None or str(value).strip() == "":
        return "Not provided"

    text = str(value).strip()

    if any(
        unit in text.lower()
        for unit in ["day", "week", "month", "hour", "year"]
    ):
        return safe_text(text)

    return safe_text(f"{text} day(s)")


def section_heading(title, styles):
    return Table(
        [
            [
                Paragraph(
                    safe_text(title),
                    styles["section_heading"],
                )
            ]
        ],
        colWidths=[174 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BLUE),
                ("BOX", (0, 0), (-1, -1), 0.7, LIGHT_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        ),
    )


def key_value_table(rows, styles):
    formatted_rows = []

    for label, value in rows:
        formatted_rows.append(
            [
                Paragraph(
                    f"<b>{safe_text(label)}</b>",
                    styles["table_label"],
                ),
                Paragraph(
                    safe_text(value),
                    styles["table_value"],
                ),
            ]
        )

    table = Table(
        formatted_rows,
        colWidths=[52 * mm, 122 * mm],
        repeatRows=0,
    )

    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
                ("BACKGROUND", (0, 0), (0, -1), VERY_LIGHT_BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    return table


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="report_title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=2,
        )
    )

    styles.add(
        ParagraphStyle(
            name="report_subtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            textColor=colors.white,
            alignment=TA_CENTER,
        )
    )

    styles.add(
        ParagraphStyle(
            name="section_heading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=14,
            textColor=HOSPITAL_BLUE,
            alignment=TA_LEFT,
            spaceAfter=0,
        )
    )

    styles.add(
        ParagraphStyle(
            name="body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=DARK_TEXT,
        )
    )

    styles.add(
        ParagraphStyle(
            name="small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=11,
            textColor=MUTED_TEXT,
        )
    )

    styles.add(
        ParagraphStyle(
            name="table_label",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=12,
            textColor=DARK_TEXT,
        )
    )

    styles.add(
        ParagraphStyle(
            name="table_value",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            textColor=DARK_TEXT,
        )
    )

    styles.add(
        ParagraphStyle(
            name="diagnosis_label",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            leading=11,
            textColor=MUTED_TEXT,
            alignment=TA_CENTER,
        )
    )

    styles.add(
        ParagraphStyle(
            name="diagnosis_value",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=19,
            textColor=HOSPITAL_BLUE,
            alignment=TA_CENTER,
        )
    )

    styles.add(
        ParagraphStyle(
            name="disclaimer",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=12,
            textColor=colors.HexColor("#6B4F00"),
        )
    )

    return styles


def draw_page_footer(canvas, doc, report_id):
    canvas.saveState()

    page_width, _ = A4

    canvas.setStrokeColor(LIGHT_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(
        doc.leftMargin,
        15 * mm,
        page_width - doc.rightMargin,
        15 * mm,
    )

    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED_TEXT)

    canvas.drawString(
        doc.leftMargin,
        10 * mm,
        f"AI Medical Assistant | Report ID: {report_id}",
    )

    canvas.drawRightString(
        page_width - doc.rightMargin,
        10 * mm,
        f"Page {doc.page}",
    )

    canvas.restoreState()


def generate_medical_report(report_data):
    """
    Generate a hospital-style symptom analysis PDF.

    Parameters
    ----------
    report_data : dict
        Data stored in session["medical_report"].

    Returns
    -------
    tuple
        (BytesIO buffer, report_id)
    """

    if not isinstance(report_data, dict):
        raise ValueError("report_data must be a dictionary.")

    now = datetime.now()
    report_id = now.strftime("AI-%Y%m%d-%H%M%S")

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=23 * mm,
        title="AI Medical Examination Report",
        author="AI Medical Assistant",
        subject="AI-generated medical assessment report",
    )

    styles = build_styles()
    story = []

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    header = Table(
        [
            [
                Paragraph(
                    "AI MEDICAL ASSISTANT",
                    styles["report_title"],
                )
            ],
            [
                Paragraph(
                    "Medical Examination Report",
                    styles["report_subtitle"],
                )
            ],
        ],
        colWidths=[174 * mm],
    )

    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HOSPITAL_BLUE),
                ("BOX", (0, 0), (-1, -1), 1, HOSPITAL_BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, 0), 13),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("TOPPADDING", (0, 1), (-1, 1), 0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
            ]
        )
    )

    story.append(header)
    story.append(Spacer(1, 6 * mm))

    metadata_table = Table(
        [
            [
                Paragraph(
                    f"<b>Report ID:</b> {safe_text(report_id)}",
                    styles["small"],
                ),
                Paragraph(
                    f"<b>Generated:</b> "
                    f"{safe_text(now.strftime('%d %B %Y, %I:%M %p'))}",
                    styles["small"],
                ),
            ]
        ],
        colWidths=[87 * mm, 87 * mm],
    )

    metadata_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    story.append(metadata_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Patient information
    # -----------------------------------------------------

    story.append(section_heading("PATIENT INFORMATION", styles))
    story.append(Spacer(1, 2 * mm))

    patient_rows = [
        ("Age", report_data.get("age")),
        ("Gender", report_data.get("gender")),
        ("Symptom Duration", format_duration(report_data.get("duration"))),
        ("Reported Severity", report_data.get("severity")),
        ("Temperature", format_temperature(report_data.get("temperature"))),
    ]

    story.append(key_value_table(patient_rows, styles))
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Symptoms
    # -----------------------------------------------------

    story.append(section_heading("REPORTED SYMPTOMS", styles))
    story.append(Spacer(1, 2 * mm))

    symptoms = report_data.get("selected_symptoms") or []

    if isinstance(symptoms, str):
        symptoms = [
            symptom.strip()
            for symptom in symptoms.split(",")
            if symptom.strip()
        ]

    if symptoms:
        symptom_rows = []

        for index, symptom in enumerate(symptoms, start=1):
            symptom_rows.append(
                [
                    Paragraph(
                        str(index),
                        styles["table_label"],
                    ),
                    Paragraph(
                        safe_text(str(symptom).replace("_", " ").title()),
                        styles["table_value"],
                    ),
                ]
            )

        symptoms_table = Table(
            symptom_rows,
            colWidths=[12 * mm, 162 * mm],
        )

        symptoms_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.4, LIGHT_BORDER),
                    ("BACKGROUND", (0, 0), (0, -1), VERY_LIGHT_BLUE),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(symptoms_table)

    else:
        story.append(
            Paragraph(
                "No symptoms were recorded.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Follow-up status
    # -----------------------------------------------------

    story.append(section_heading("FOLLOW-UP ASSESSMENT", styles))
    story.append(Spacer(1, 2 * mm))

    refinement_used = bool(report_data.get("refinement_used"))
    follow_up_answers = report_data.get("follow_up_answers") or {}

    if refinement_used and follow_up_answers:
        follow_up_rows = []

        for question, answer in follow_up_answers.items():
            readable_question = str(question).replace("_", " ").title()

            follow_up_rows.append(
                (
                    readable_question,
                    answer,
                )
            )

        story.append(key_value_table(follow_up_rows, styles))

    elif refinement_used:
        story.append(
            Paragraph(
                "Additional follow-up information was used to refine "
                "the initial prediction.",
                styles["body"],
            )
        )

    else:
        story.append(
            Paragraph(
                "The initial prediction confidence did not require "
                "additional follow-up refinement.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Diagnosis summary
    # -----------------------------------------------------

    story.append(section_heading("FINAL AI ASSESSMENT", styles))
    story.append(Spacer(1, 2 * mm))

    diagnosis_table = Table(
        [
            [
                Paragraph(
                    "PREDICTED CONDITION",
                    styles["diagnosis_label"],
                ),
                Paragraph(
                    "CONFIDENCE",
                    styles["diagnosis_label"],
                ),
            ],
            [
                Paragraph(
                    safe_text(
                        report_data.get("final_prediction"),
                        "Not available",
                    ),
                    styles["diagnosis_value"],
                ),
                Paragraph(
                    format_confidence(
                        report_data.get("final_confidence")
                    ),
                    styles["diagnosis_value"],
                ),
            ],
        ],
        colWidths=[116 * mm, 58 * mm],
    )

    diagnosis_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), SUCCESS_BACKGROUND),
                ("BOX", (0, 0), (-1, -1), 1, SUCCESS_BORDER),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, SUCCESS_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
                ("TOPPADDING", (0, 1), (-1, 1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 11),
            ]
        )
    )

    story.append(diagnosis_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Top predictions
    # -----------------------------------------------------

    story.append(section_heading("TOP PREDICTIONS", styles))
    story.append(Spacer(1, 2 * mm))

    predictions = report_data.get("top_predictions") or []

    prediction_rows = [
        [
            Paragraph("<b>Rank</b>", styles["table_label"]),
            Paragraph("<b>Condition</b>", styles["table_label"]),
            Paragraph("<b>Confidence</b>", styles["table_label"]),
        ]
    ]

    for index, prediction in enumerate(predictions[:3], start=1):
        if not isinstance(prediction, dict):
            continue

        prediction_rows.append(
            [
                Paragraph(str(index), styles["table_value"]),
                Paragraph(
                    safe_text(prediction.get("disease"), "Unknown"),
                    styles["table_value"],
                ),
                Paragraph(
                    format_confidence(prediction.get("confidence")),
                    styles["table_value"],
                ),
            ]
        )

    if len(prediction_rows) == 1:
        prediction_rows.append(
            [
                Paragraph("-", styles["table_value"]),
                Paragraph(
                    "No prediction details available.",
                    styles["table_value"],
                ),
                Paragraph("-", styles["table_value"]),
            ]
        )

    predictions_table = Table(
        prediction_rows,
        colWidths=[20 * mm, 114 * mm, 40 * mm],
        repeatRows=1,
    )

    predictions_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HOSPITAL_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 1), (0, -1), "CENTER"),
                ("ALIGN", (2, 1), (2, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    VERY_LIGHT_BLUE,
                ]),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(predictions_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Medicine recommendations
    # -----------------------------------------------------

    story.append(section_heading("MEDICINE RECOMMENDATIONS", styles))
    story.append(Spacer(1, 2 * mm))

    medicines = report_data.get("medicine_recommendations") or []
    medicine_warning = report_data.get("medicine_warning")
    medicine_confidence_threshold = report_data.get(
        "medicine_confidence_threshold",
        55,
    )

    medicine_rows = [
        [
            Paragraph("<b>Medicine</b>", styles["table_label"]),
            Paragraph("<b>Dosage / Instructions</b>", styles["table_label"]),
        ]
    ]

    if isinstance(medicines, dict):
        medicines = [medicines]

    for medicine in medicines:
        if isinstance(medicine, dict):
            medicine_name = (
                medicine.get("medicine")
                or medicine.get("name")
                or medicine.get("Medicine")
                or "Not specified"
            )

            dosage = (
                medicine.get("dosage")
                or medicine.get("dose")
                or medicine.get("Dosage")
                or "Follow medical guidance"
            )

        else:
            medicine_name = medicine
            dosage = "Follow medical guidance"

        medicine_rows.append(
            [
                Paragraph(
                    safe_text(medicine_name),
                    styles["table_value"],
                ),
                Paragraph(
                    safe_text(dosage),
                    styles["table_value"],
                ),
            ]
        )

    if len(medicine_rows) == 1:

        if medicine_warning:
            confidence_text = format_confidence(
                report_data.get("final_confidence")
            )

            threshold_text = format_confidence(
                medicine_confidence_threshold
            )

            warning_text = (
                f"{safe_text(medicine_warning)} "
                f"Prediction confidence: {confidence_text}. "
                f"Required confidence: {threshold_text}."
            )

            medicine_rows.append(
                [
                    Paragraph(
                        "Recommendations withheld",
                        styles["table_value"],
                    ),
                    Paragraph(
                        warning_text,
                        styles["table_value"],
                    ),
                ]
            )

        else:
            medicine_rows.append(
                [
                    Paragraph(
                        "No medicine recommendation available.",
                        styles["table_value"],
                    ),
                    Paragraph(
                        "Consult a qualified medical professional.",
                        styles["table_value"],
                    ),
                ]
            )

    medicine_table = Table(
        medicine_rows,
        colWidths=[75 * mm, 99 * mm],
        repeatRows=1,
    )

    medicine_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HOSPITAL_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    VERY_LIGHT_BLUE,
                ]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(medicine_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Disease description
    # -----------------------------------------------------

    story.append(section_heading("CONDITION DESCRIPTION", styles))
    story.append(Spacer(1, 2 * mm))

    story.append(
        Table(
            [
                [
                    Paragraph(
                        safe_text(
                            report_data.get("description"),
                            "No description is currently available.",
                        ),
                        styles["body"],
                    )
                ]
            ],
            colWidths=[174 * mm],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            ),
        )
    )

    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------

    story.append(section_heading("IMPORTANT MEDICAL DISCLAIMER", styles))
    story.append(Spacer(1, 2 * mm))

    disclaimer_text = (
        "This report was generated by an artificial intelligence system "
        "for educational and decision-support purposes only. It does not "
        "constitute a confirmed medical diagnosis, prescription, or "
        "professional medical advice. Medicine recommendations must not "
        "be used without consultation with a qualified healthcare "
        "professional. Seek immediate medical assistance if symptoms are "
        "severe, rapidly worsening, or life-threatening."
    )

    disclaimer_table = Table(
        [
            [
                Paragraph(
                    disclaimer_text,
                    styles["disclaimer"],
                )
            ]
        ],
        colWidths=[174 * mm],
    )

    disclaimer_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WARNING_BACKGROUND),
                ("BOX", (0, 0), (-1, -1), 0.8, WARNING_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )

    story.append(disclaimer_table)
    story.append(Spacer(1, 5 * mm))

    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=LIGHT_BORDER,
        )
    )

    story.append(Spacer(1, 2 * mm))

    story.append(
        Paragraph(
            "End of AI-generated medical examination report.",
            styles["small"],
        )
    )

    doc.build(
        story,
        onFirstPage=lambda canvas, document: draw_page_footer(
            canvas,
            document,
            report_id,
        ),
        onLaterPages=lambda canvas, document: draw_page_footer(
            canvas,
            document,
            report_id,
        ),
    )

    buffer.seek(0)

    return buffer, report_id

def generate_image_analysis_report(report_data):
    """
    Generate a PDF report for either the skin-disease or chest X-ray module.

    Expected report_data keys:
        module_name
        prediction
        confidence
        description
        medicines
        precautions
        image_path
        confidence_threshold
        low_confidence
    """
    if not isinstance(report_data, dict):
        raise ValueError("report_data must be a dictionary.")

    now = datetime.now()
    report_id = now.strftime("AI-%Y%m%d-%H%M%S")
    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=23 * mm,
        title="AI Medical Image Analysis Report",
        author="AI Medical Assistant",
        subject="AI-generated medical image analysis report",
    )

    styles = build_styles()
    story = []

    module_name = safe_text(
        report_data.get("module_name"),
        "Medical Image Analysis",
    )

    prediction = safe_text(
        report_data.get("prediction"),
        "Not available",
    )

    confidence = report_data.get("confidence")
    confidence_text = format_confidence(confidence)

    low_confidence = bool(report_data.get("low_confidence"))
    confidence_threshold = report_data.get("confidence_threshold", 60)

    # -----------------------------------------------------
    # Header
    # -----------------------------------------------------

    header = Table(
        [
            [
                Paragraph(
                    "AI MEDICAL ASSISTANT",
                    styles["report_title"],
                )
            ],
            [
                Paragraph(
                    f"{module_name} Report",
                    styles["report_subtitle"],
                )
            ],
        ],
        colWidths=[174 * mm],
    )

    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), HOSPITAL_BLUE),
                ("BOX", (0, 0), (-1, -1), 1, HOSPITAL_BLUE),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, 0), 13),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
                ("TOPPADDING", (0, 1), (-1, 1), 0),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
            ]
        )
    )

    story.append(header)
    story.append(Spacer(1, 6 * mm))

    metadata_table = Table(
        [
            [
                Paragraph(
                    f"<b>Report ID:</b> {safe_text(report_id)}",
                    styles["small"],
                ),
                Paragraph(
                    f"<b>Generated:</b> "
                    f"{safe_text(now.strftime('%d %B %Y, %I:%M %p'))}",
                    styles["small"],
                ),
            ]
        ],
        colWidths=[87 * mm, 87 * mm],
    )

    metadata_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    story.append(metadata_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Uploaded image
    # -----------------------------------------------------

    image_path = report_data.get("image_path")

    if image_path:
        try:
            report_image = RLImage(
                image_path,
                width=75 * mm,
                height=75 * mm,
            )

            image_table = Table(
                [[report_image]],
                colWidths=[174 * mm],
            )

            image_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BOX", (0, 0), (-1, -1), 0.6, LIGHT_BORDER),
                        ("BACKGROUND", (0, 0), (-1, -1), VERY_LIGHT_BLUE),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                )
            )

            story.append(section_heading("UPLOADED IMAGE", styles))
            story.append(Spacer(1, 2 * mm))
            story.append(image_table)
            story.append(Spacer(1, 5 * mm))

        except Exception:
            # The report should still generate even if the image cannot be embedded.
            pass

    # -----------------------------------------------------
    # AI assessment
    # -----------------------------------------------------

    story.append(section_heading("AI ASSESSMENT", styles))
    story.append(Spacer(1, 2 * mm))

    assessment_table = Table(
        [
            [
                Paragraph(
                    "PREDICTION",
                    styles["diagnosis_label"],
                ),
                Paragraph(
                    "CONFIDENCE",
                    styles["diagnosis_label"],
                ),
            ],
            [
                Paragraph(
                    prediction,
                    styles["diagnosis_value"],
                ),
                Paragraph(
                    confidence_text,
                    styles["diagnosis_value"],
                ),
            ],
        ],
        colWidths=[116 * mm, 58 * mm],
    )

    assessment_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    WARNING_BACKGROUND if low_confidence else SUCCESS_BACKGROUND,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    1,
                    WARNING_BORDER if low_confidence else SUCCESS_BORDER,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    WARNING_BORDER if low_confidence else SUCCESS_BORDER,
                ),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 3),
                ("TOPPADDING", (0, 1), (-1, 1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 11),
            ]
        )
    )

    story.append(assessment_table)
    story.append(Spacer(1, 4 * mm))

    if low_confidence:
        warning_message = (
            "The model confidence is below the required "
            f"{safe_text(confidence_threshold)}% threshold. "
            "This prediction should be treated as uncertain. "
            "No medicine recommendation has been provided. "
            "Please consult a qualified healthcare professional."
        )

        low_conf_table = Table(
            [[Paragraph(warning_message, styles["disclaimer"])]],
            colWidths=[174 * mm],
        )

        low_conf_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), WARNING_BACKGROUND),
                    ("BOX", (0, 0), (-1, -1), 0.8, WARNING_BORDER),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            )
        )

        story.append(low_conf_table)
        story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Description
    # -----------------------------------------------------

    story.append(section_heading("CONDITION DESCRIPTION", styles))
    story.append(Spacer(1, 2 * mm))

    description = safe_text(
        report_data.get("description"),
        "No description is currently available.",
    )

    story.append(
        Table(
            [[Paragraph(description, styles["body"])]],
            colWidths=[174 * mm],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 9),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ]
            ),
        )
    )

    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Medicine recommendations
    # -----------------------------------------------------

    story.append(section_heading("MEDICINE RECOMMENDATIONS", styles))
    story.append(Spacer(1, 2 * mm))

    medicines = report_data.get("medicines") or []

    medicine_rows = [
        [
            Paragraph("<b>Medicine</b>", styles["table_label"]),
            Paragraph(
                "<b>Dosage / Instructions</b>",
                styles["table_label"],
            ),
        ]
    ]

    if not low_confidence:
        for medicine in medicines:
            if isinstance(medicine, (list, tuple)) and len(medicine) >= 2:
                medicine_name = medicine[0]
                dosage = medicine[1]
            elif isinstance(medicine, dict):
                medicine_name = (
                    medicine.get("medicine")
                    or medicine.get("name")
                    or "Not specified"
                )
                dosage = (
                    medicine.get("dosage")
                    or medicine.get("dose")
                    or "Follow medical guidance"
                )
            else:
                medicine_name = medicine
                dosage = "Follow medical guidance"

            medicine_rows.append(
                [
                    Paragraph(
                        safe_text(medicine_name),
                        styles["table_value"],
                    ),
                    Paragraph(
                        safe_text(dosage),
                        styles["table_value"],
                    ),
                ]
            )

    if len(medicine_rows) == 1:
        if low_confidence:
            medicine_rows.append(
                [
                    Paragraph(
                        "Recommendations withheld",
                        styles["table_value"],
                    ),
                    Paragraph(
                        "Confidence is below the required threshold. "
                        "Consult a qualified healthcare professional.",
                        styles["table_value"],
                    ),
                ]
            )
        else:
            medicine_rows.append(
                [
                    Paragraph(
                        "No medicine required / available",
                        styles["table_value"],
                    ),
                    Paragraph(
                        "No disease-specific medicine recommendation "
                        "is provided for this result.",
                        styles["table_value"],
                    ),
                ]
            )

    medicine_table = Table(
        medicine_rows,
        colWidths=[75 * mm, 99 * mm],
        repeatRows=1,
    )

    medicine_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HOSPITAL_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, LIGHT_BORDER),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    VERY_LIGHT_BLUE,
                ]),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )

    story.append(medicine_table)
    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Precautions
    # -----------------------------------------------------

    story.append(section_heading("PRECAUTIONS / GUIDANCE", styles))
    story.append(Spacer(1, 2 * mm))

    precautions = report_data.get("precautions") or []

    if precautions:
        precaution_rows = []

        for index, precaution in enumerate(precautions, start=1):
            precaution_rows.append(
                [
                    Paragraph(
                        str(index),
                        styles["table_label"],
                    ),
                    Paragraph(
                        safe_text(precaution),
                        styles["table_value"],
                    ),
                ]
            )

        precaution_table = Table(
            precaution_rows,
            colWidths=[12 * mm, 162 * mm],
        )

        precaution_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.4, LIGHT_BORDER),
                    ("BACKGROUND", (0, 0), (0, -1), VERY_LIGHT_BLUE),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        story.append(precaution_table)

    else:
        story.append(
            Paragraph(
                "No additional precautions are available.",
                styles["body"],
            )
        )

    story.append(Spacer(1, 5 * mm))

    # -----------------------------------------------------
    # Disclaimer
    # -----------------------------------------------------

    story.append(section_heading("IMPORTANT MEDICAL DISCLAIMER", styles))
    story.append(Spacer(1, 2 * mm))

    disclaimer_text = (
        "This report was generated by an artificial intelligence system "
        "for educational and decision-support purposes only. It does not "
        "constitute a confirmed medical diagnosis, prescription, or "
        "professional medical advice. Medicine recommendations must not "
        "be used without consultation with a qualified healthcare "
        "professional. Seek prompt medical attention if symptoms are "
        "severe, persistent, or worsening."
    )

    disclaimer_table = Table(
        [[Paragraph(disclaimer_text, styles["disclaimer"])]],
        colWidths=[174 * mm],
    )

    disclaimer_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), WARNING_BACKGROUND),
                ("BOX", (0, 0), (-1, -1), 0.8, WARNING_BORDER),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )

    story.append(disclaimer_table)

    doc.build(
        story,
        onFirstPage=lambda canvas, document: draw_page_footer(
            canvas,
            document,
            report_id,
        ),
        onLaterPages=lambda canvas, document: draw_page_footer(
            canvas,
            document,
            report_id,
        ),
    )

    buffer.seek(0)
    return buffer, report_id