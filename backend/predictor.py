"""Loads the saved artifacts and serves ensemble predictions."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn

import config
from backend.data_loader import (
    ReferenceData, load_reference_data, match_key, severity_lookup, symptom_label,
)
from backend.ensemble import full_proba, soft_vote


class ModelArtifactError(RuntimeError):
    """Model files are missing or inconsistent. Messages never include file-system paths."""


def _symptoms_to_row(symptoms, index: dict[str, int], n_features: int) -> np.ndarray:
    vec = np.zeros((1, n_features), dtype=np.uint8)
    for s in symptoms:
        vec[0, index[s]] = 1
    return vec


class DiseasePredictor:
    def __init__(self, models_dir: Path | str = config.MODELS_DIR):
        models_dir = Path(models_dir)
        needed = [f"{k}.pkl" for k in config.MODEL_KEYS] + [
            "ensemble_config.pkl", "label_encoder.pkl", "symptom_features.pkl"]
        missing = [n for n in needed if not (models_dir / n).is_file()]
        if missing:
            raise ModelArtifactError(
                "Model artifacts are missing (" + ", ".join(missing) + "). "
                "Run `python -m training.train` to create them.")
        try:
            self.models = {k: joblib.load(models_dir / f"{k}.pkl") for k in config.MODEL_KEYS}
            self.label_encoder = joblib.load(models_dir / "label_encoder.pkl")
            self.vocabulary: list[str] = joblib.load(models_dir / "symptom_features.pkl")
            self.ensemble_config: dict = joblib.load(models_dir / "ensemble_config.pkl")
        except Exception as exc:
            raise ModelArtifactError(
                "Model artifacts could not be loaded; retrain with `python -m training.train`.") from exc

        self.classes: list[str] = list(self.label_encoder.classes_)
        self.n_classes = len(self.classes)
        self.index = {s: i for i, s in enumerate(self.vocabulary)}
        cfg = self.ensemble_config
        if cfg.get("n_classes") != self.n_classes or cfg.get("n_features") != len(self.vocabulary):
            raise ModelArtifactError("Model artifacts are inconsistent; retrain the models.")
        self.mode = cfg["selected_mode"]
        self.weights = cfg["weights"][self.mode]
        self.version_warning = None
        if cfg.get("sklearn_version") != sklearn.__version__:
            self.version_warning = "Models were trained with a different scikit-learn version."
            warnings.warn(self.version_warning)
        self.reference: ReferenceData = load_reference_data(self.classes)
        self._disease_by_key = {match_key(c): c for c in self.classes}

    # -- catalogue ---------------------------------------------------------- #
    def symptom_catalog(self) -> list[dict]:
        sev = severity_lookup(self.reference, self.vocabulary)
        return [{"name": s, "label": symptom_label(s), "severity": sev[s]} for s in self.vocabulary]

    def disease_info(self, name: str) -> dict | None:
        disease = self._disease_by_key.get(match_key(name))
        if disease is None:
            return None
        return {"disease": disease, "description": self.reference.description_for(disease),
                "precautions": self.reference.precautions_for(disease)}

    # -- prediction --------------------------------------------------------- #
    def model_probabilities(self, symptoms: list[str]) -> dict[str, np.ndarray]:
        X = _symptoms_to_row(symptoms, self.index, len(self.vocabulary))
        return {k: full_proba(m, X, self.n_classes)[0:1] for k, m in self.models.items()}

    def predict(self, symptoms: list[str]) -> dict:
        """`symptoms` must already be validated, normalised names from the vocabulary."""
        probas = self.model_probabilities(symptoms)
        final = soft_vote(probas, self.weights)[0]
        if not np.isfinite(final).all():
            raise RuntimeError("non-finite probabilities")
        order = np.argsort(-final)
        top = int(order[0])

        details, votes = {}, {}
        for k, p in probas.items():
            p = p[0]
            best = int(p.argmax())
            votes[k] = self.classes[best]
            details[k] = {
                "label": config.MODEL_LABELS[k],
                "prediction": self.classes[best],
                "confidence": float(p[best]),
                "probability_for_ensemble_choice": float(p[top]),
                "weight": float(self.weights[k]),
            }
        agree = sum(1 for v in votes.values() if v == self.classes[top])
        confidence = float(final[top])

        warnings_out = []
        if len(symptoms) < 3:
            warnings_out.append("Every record the models learned from lists at least 3 symptoms; "
                                "predictions from fewer than 3 are less reliable.")
        if confidence < config.LOW_CONFIDENCE_THRESHOLD:
            warnings_out.append("Ensemble confidence is low. Several diseases fit these symptoms; "
                                "see the alternatives below.")
        if agree < len(votes):
            warnings_out.append(f"The models disagree: {agree} of {len(votes)} chose the top result.")

        sev = severity_lookup(self.reference, symptoms)
        known = [v for v in sev.values() if v is not None]
        info = self.disease_info(self.classes[top]) or {}
        return {
            "predicted_disease": self.classes[top],
            "ensemble_confidence": confidence,
            "ensemble_method": "soft voting (weighted average of class probabilities)",
            "ensemble_mode": self.mode,
            "model_predictions": votes,
            "model_details": details,
            "top_predictions": [{"disease": self.classes[int(i)], "probability": float(final[int(i)])}
                                for i in order[:5]],
            "models_agreeing": agree,
            "description": info.get("description"),
            "precautions": info.get("precautions", []),
            "selected_symptoms": [{"name": s, "label": symptom_label(s), "severity": sev[s]} for s in symptoms],
            "severity_summary": {"total": int(sum(known)), "max": max(known) if known else None,
                                 "note": "Severity weights are shown for context only; they are not model inputs."},
            "warnings": warnings_out,
            "disclaimer": config.DISCLAIMER,
        }
