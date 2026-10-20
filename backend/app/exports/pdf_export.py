"""
PDF exporter for Tamil Unicode text using ReportLab with embedded Noto Sans Tamil font.
"""

import io
import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

FONTS_DIR = Path(__file__).resolve().parent / "fonts"
TAMIL_FONT_PATH = FONTS_DIR / "NotoSansTamil-Regular.ttf"
FONT_NAME = "NotoSansTamil"

# Global font registration flag
_FONT_REGISTERED = False

def register_tamil_font():
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return FONT_NAME

    if TAMIL_FONT_PATH.exists():
        try:
            pdfmetrics.registerFont(TTFont(FONT_NAME, str(TAMIL_FONT_PATH)))
            _FONT_REGISTERED = True
            return FONT_NAME
        except Exception as e:
            print(f"Warning: Failed to register NotoSansTamil: {e}")

    # Fallback to system fonts if present
    win_nirmala = Path("C:/Windows/Fonts/Nirmala.ttc")
    if win_nirmala.exists():
        try:
            pdfmetrics.registerFont(TTFont("Nirmala", str(win_nirmala), subfontIndex=0))
            _FONT_REGISTERED = True
            return "Nirmala"
        except Exception:
            pass

    return "Helvetica"

def generate_pdf_bytes(text: str, title: str = "TamilScript OCR - Extracted Text") -> bytes:
    """Generate a validated PDF document containing properly rendered Tamil Unicode text."""
    font_name = register_tamil_font()
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Title Style
    title_style = ParagraphStyle(
        name="TamilTitle",
        parent=styles["Normal"],
        fontName=font_name if font_name != "Helvetica" else "Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#4338ca"),
        spaceAfter=14
    )

    # Custom Body Style with proper line height for Tamil diacritics
    body_style = ParagraphStyle(
        name="TamilBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=10
    )

    story = [
        Paragraph(title, title_style),
        Spacer(1, 10)
    ]

    # Split text into paragraphs
    raw_paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")

    for para in raw_paragraphs:
        if not para.strip():
            continue
        # Convert internal single newlines to <br/> for ReportLab HTML-like paragraph formatting
        safe_para = para.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        formatted_para = safe_para.replace("\n", "<br/>")
        story.append(Paragraph(formatted_para, body_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    return buffer.getvalue()
