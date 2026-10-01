"""Flask application: REST API + static frontend.

Run:  python app.py        (after `python -m training.train`)
"""
from __future__ import annotations

import json
import logging

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

import config
from backend.data_loader import DatasetError
from backend.predictor import DiseasePredictor, ModelArtifactError
from backend.validation import ValidationError, validate_payload

log = logging.getLogger("disease_prediction_ai")

CSP = ("default-src 'self'; img-src 'self' data:; style-src 'self' https://fonts.googleapis.com; "
       "font-src https://fonts.gstatic.com; script-src 'self'; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'none'; form-action 'self'")


def _err(message: str, status: int, field: str | None = None):
    body = {"error": message}
    if field:
        body["field"] = field
    return jsonify(body), status


def create_app(predictor: DiseasePredictor | None = None, load_models: bool = True) -> Flask:
    app = Flask(__name__, static_folder=str(config.FRONTEND_DIR), static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
    app.json.sort_keys = False

    state = {"predictor": predictor, "error": None}
    if predictor is None and load_models:
        try:
            state["predictor"] = DiseasePredictor()
        except (ModelArtifactError, DatasetError) as exc:
            state["error"] = str(exc)
            log.error("Model loading failed: %s", exc)

    def get_predictor():
        if state["predictor"] is None:
            raise ModelArtifactError(state["error"] or "Models are not loaded. Run `python -m training.train` first.")
        return state["predictor"]

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return send_from_directory(config.FRONTEND_DIR, "index.html")

    # -------------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        ready = state["predictor"] is not None
        return jsonify({"status": "ok" if ready else "degraded", "models_loaded": ready,
                        "message": None if ready else state["error"]}), (200 if ready else 503)

    @app.get("/api/symptoms")
    def symptoms():
        cat = get_predictor().symptom_catalog()
        return jsonify({"count": len(cat), "symptoms": cat})

    @app.get("/api/disease/<disease>")
    def disease(disease):
        if len(disease) > 120:
            return _err("Disease name too long", 400)
        info = get_predictor().disease_info(disease)
        if info is None:
            return _err("Disease not found", 404)
        return jsonify(info)

    @app.post("/api/predict")
    def predict():
        pred = get_predictor()
        if not request.is_json:
            return _err("Content-Type must be application/json", 415)
        payload = request.get_json(silent=True)
        if payload is None:
            return _err("Request body is empty or not valid JSON", 400)
        data = validate_payload(payload, set(pred.vocabulary))
        try:
            result = pred.predict(data["symptoms"])
        except Exception:
            log.exception("Prediction failed")
            return _err("Prediction failed. Please try again.", 500)
        if not result:
            return _err("The model returned an empty result", 500)
        result["patient_info"] = {"age": data["age"], "gender": data["gender"], **data["additional_info"]}
        return jsonify(result)

    def _metrics():
        path = config.REPORTS_DIR / "metrics.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @app.get("/api/model-performance")
    def model_performance():
        data = _metrics()
        if data is None:
            return _err("Evaluation results not found. Run `python -m training.train`.", 503)
        return jsonify(data)

    @app.get("/api/report-image/<name>")
    def report_image(name):
        data = _metrics()
        if data is None or name not in data.get("images", []):
            return _err("Image not found", 404)
        return send_from_directory(config.REPORTS_DIR, f"{name}.png", mimetype="image/png")

    # ----------------------------------------------------------- error handling
    @app.errorhandler(ValidationError)
    def on_validation(exc):
        return _err(exc.message, 400, exc.field)

    @app.errorhandler(ModelArtifactError)
    def on_models(exc):
        return _err(str(exc), 503)

    @app.errorhandler(HTTPException)
    def on_http(exc):
        messages = {413: "Request body too large", 404: "Not found", 405: "Method not allowed"}
        return _err(messages.get(exc.code, exc.name), exc.code or 500)

    @app.errorhandler(Exception)
    def on_unexpected(exc):
        log.exception("Unhandled error")
        return _err("Internal server error", 500)

    @app.after_request
    def security_headers(resp):
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/") and not request.path.startswith("/api/report-image/"):
            resp.headers.setdefault("Cache-Control", "no-store")
        return resp

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Open http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
