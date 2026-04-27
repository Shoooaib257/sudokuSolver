from flask import Flask, jsonify
from app.routes.sudoku_routes import sudoku_bp

def create_app():
    app = Flask(__name__)

    # Root health check
    @app.route("/")
    def index():
        return jsonify({
            "name": "Sudoku Solver API",
            "version": "1.0",
            "status": "active"
        })

    # Register blueprints with prefix for better versioning/grouping
    app.register_blueprint(sudoku_bp, url_prefix="/api/sudoku")

    return app

if __name__ == "__main__":
    app = create_app()
    # In production, use a proper WSGI server like gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
