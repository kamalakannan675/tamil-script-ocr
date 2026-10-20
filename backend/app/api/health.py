from flask import Blueprint, jsonify
import sys
import platform

health_bp = Blueprint("health", __name__)

@health_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "TamilScript OCR Platform",
        "version": "1.0.0",
        "python_version": sys.version.split()[0],
        "platform": platform.platform()
    })
