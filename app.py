from flask import Flask
from blueprints.main import main_bp
from blueprints.ndvi import ndvi_bp
from blueprints.synthetic import synthetic_bp
from blueprints.forecast import forecast_bp
from blueprints.datasets import datasets_bp


def create_app() -> Flask:
    app = Flask(__name__)

    app.register_blueprint(main_bp)
    app.register_blueprint(ndvi_bp)
    app.register_blueprint(synthetic_bp)
    app.register_blueprint(forecast_bp)
    app.register_blueprint(datasets_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)
