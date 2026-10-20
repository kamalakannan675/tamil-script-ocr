"""
DOCX exporter for Tamil Unicode text with proper font attributes.
"""

import io
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def generate_docx_bytes(text: str, title: str = "TamilScript OCR - Extracted Text") -> bytes:
    """Generate a validated Word (.docx) file containing the Tamil Unicode text."""
    doc = Document()

    # Configure sensible page margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Title
    title_p = doc.add_paragraph()
    title_run = title_p.add_run(title)
    title_run.font.name = "Arial"
    title_run.font.size = Pt(14)
    title_run.font.bold = True
    title_run.font.color.rgb = RGBColor(79, 70, 229)  # Brand indigo
    title_p.paragraph_format.space_after = Pt(14)

    # Split into paragraphs while preserving newlines
    paragraphs = text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n")

    for para_text in paragraphs:
        if not para_text.strip():
            continue
            
        p = doc.add_paragraph()
        lines = para_text.split("\n")
        for i, line in enumerate(lines):
            run = p.add_run(line)
            # Set font to cross-platform Tamil font preferences
            run.font.name = "Noto Sans Tamil"
            run.font.size = Pt(11)
            # In docx, add break between lines in the same paragraph
            if i < len(lines) - 1:
                p.add_run("\n")
                
        p.paragraph_format.line_spacing = 1.2
        p.paragraph_format.space_after = Pt(8)

    output = io.BytesIO()
    doc.save(output)
    return output.getvalue()
