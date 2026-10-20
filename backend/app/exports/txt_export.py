"""
TXT exporter for Tamil Unicode text.
"""

def generate_txt_bytes(text: str) -> bytes:
    """Return UTF-8 encoded bytes of text with proper line breaks."""
    clean_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return clean_text.encode("utf-8")
