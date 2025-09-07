from flask import Flask

from .config import Config


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)

    # Defer imports to avoid circulars
    from .routes import api_bp

    app.register_blueprint(api_bp, url_prefix="/api/v1")
    return app
