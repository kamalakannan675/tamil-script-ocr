"""
Local development server for TamilScript OCR backend.
"""

import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() in ("true", "1")
    print(f"Starting TamilScript OCR Backend on http://127.0.0.1:{port} (Debug: {debug})...")
    app.run(host="127.0.0.1", port=port, debug=debug)
