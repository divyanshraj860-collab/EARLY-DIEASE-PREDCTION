"""Request validation and sanitisation. Nothing from the client is ever executed or
used as a path; free text is stripped of control characters and angle brackets."""
from __future__ import annotations

import difflib
import re

import config
from backend.data_loader import normalize_symptom

GENDERS = ["Male", "Female", "Other", "Prefer not to say"]
BLOOD_GROUPS = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-", "Unknown"]
SMOKING = ["Never", "Former", "Current"]
ALCOHOL = ["None", "Occasional", "Regular"]
TEXT_FIELDS = {"existing_conditions": 400, "allergies": 400, "medications": 400, "family_history": 400}


class ValidationError(ValueError):
    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.message, self.field = message, field


def clean_text(value, max_len: int = 400, field: str | None = None) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    if not isinstance(value, str):
        raise ValidationError("must be text", field)
    value = re.sub(r"[\x00-\x1f\x7f<>]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > max_len:
        raise ValidationError(f"must be at most {max_len} characters", field)
    return value


def _suggest(name: str, known) -> list[str]:
    """Close matches for an unknown symptom, e.g. 'fever' -> high_fever, mild_fever."""
    known = sorted(known)
    hits = [k for k in known if name in k][:3]
    return hits or difflib.get_close_matches(name, known, n=3, cutoff=0.6)


def _number(value, field: str, lo: float, hi: float, required: bool = False):
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValidationError(f"{field} is required", field)
        return None
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be a number", field)
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number", field) from None
    if num != num or num in (float("inf"), float("-inf")) or not (lo <= num <= hi):
        raise ValidationError(f"{field} must be between {lo:g} and {hi:g}", field)
    return int(num) if num.is_integer() else round(num, 1)


def _choice(value, field: str, options: list[str]):
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str) or value.strip() not in options:
        raise ValidationError(f"{field} must be one of: {', '.join(options)}", field)
    return value.strip()


def validate_payload(payload, known_symptoms) -> dict:
    """Validate a /api/predict body. Returns clean data or raises ValidationError."""
    if not isinstance(payload, dict):
        raise ValidationError("Request body must be a JSON object")

    age = _number(payload.get("age"), "age", 0, 120, required=True)
    if age <= 0: # type: ignore
        raise ValidationError("age must be greater than 0", "age")
    gender = _choice(payload.get("gender"), "gender", GENDERS)
    if gender is None:
        raise ValidationError("gender is required", "gender")

    raw = payload.get("symptoms")
    if not isinstance(raw, list):
        raise ValidationError("symptoms must be a list of symptom names", "symptoms")
    if len(raw) > config.MAX_SYMPTOMS * 2:
        raise ValidationError("too many symptoms supplied", "symptoms")
    symptoms, unknown = [], []
    for item in raw:
        if not isinstance(item, str) or len(item) > 80:
            raise ValidationError("each symptom must be a short text value", "symptoms")
        name = normalize_symptom(item)
        if not name:
            continue
        if name not in known_symptoms:
            hint = _suggest(name, known_symptoms)
            unknown.append(clean_text(item, 80) + (f" (did you mean: {', '.join(hint)}?)" if hint else ""))
        elif name not in symptoms:
            symptoms.append(name)
    if unknown:
        shown = "; ".join(unknown[:5]) + (" ..." if len(unknown) > 5 else "")
        raise ValidationError(f"Unknown symptom(s): {shown}", "symptoms")
    if not symptoms:
        raise ValidationError("Select at least one symptom", "symptoms")
    if len(symptoms) > config.MAX_SYMPTOMS:
        raise ValidationError(f"Select at most {config.MAX_SYMPTOMS} symptoms", "symptoms")

    extra = payload.get("additional_info") or {}
    if not isinstance(extra, dict):
        raise ValidationError("additional_info must be an object", "additional_info")
    info = {
        "height_cm": _number(extra.get("height"), "height", 30, 272),
        "weight_kg": _number(extra.get("weight"), "weight", 2, 650),
        "blood_group": _choice(extra.get("blood_group"), "blood_group", BLOOD_GROUPS),
        "smoking_status": _choice(extra.get("smoking_status"), "smoking_status", SMOKING),
        "alcohol_consumption": _choice(extra.get("alcohol_consumption"), "alcohol_consumption", ALCOHOL),
    }
    for key, limit in TEXT_FIELDS.items():
        val = extra.get(key)
        info[key] = clean_text(val, limit, key) if val not in (None, "", []) else None
    if info["height_cm"] and info["weight_kg"]:
        info["bmi"] = round(info["weight_kg"] / (info["height_cm"] / 100) ** 2, 1)
    return {"age": age, "gender": gender, "symptoms": symptoms,
            "additional_info": {k: v for k, v in info.items() if v is not None}}
