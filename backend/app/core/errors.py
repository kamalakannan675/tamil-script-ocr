from flask import jsonify

class APIError(Exception):
    def __init__(self, message, status_code=400, code="BAD_REQUEST", details=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}

    def to_dict(self):
        rv = {"success": False, "error": self.message, "code": self.code}
        if self.details:
            rv["details"] = self.details
        return rv

def register_error_handlers(app):
    @app.errorhandler(APIError)
    def handle_api_error(error):
        response = jsonify(error.to_dict())
        response.status_code = error.status_code
        return response

    @app.errorhandler(413)
    def handle_oversized_file(error):
        return jsonify({
            "success": False,
            "error": "File size exceeds the maximum allowed upload limit (25 MB).",
            "code": "FILE_TOO_LARGE"
        }), 413

    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({
            "success": False,
            "error": "The requested resource was not found.",
            "code": "NOT_FOUND"
        }), 404

    @app.errorhandler(500)
    def handle_internal_server_error(error):
        return jsonify({
            "success": False,
            "error": "An internal server error occurred while processing the request.",
            "code": "INTERNAL_SERVER_ERROR"
        }), 500
