from flask import Flask
from flask_cors import CORS
from .core.config import Config
from .core.errors import register_error_handlers
from .api import health_bp, ocr_bp, export_bp

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Configure CORS
    CORS(app, resources={r"/api/*": {"origins": config_class.CORS_ORIGINS}})

    # Register error handlers
    register_error_handlers(app)

    # Register API Blueprints under /api prefix
    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(ocr_bp, url_prefix="/api")
    app.register_blueprint(export_bp, url_prefix="/api")

    return app
