"""Loading and normalising the four source CSV files.

The same normalisation helpers are used at training time and at serving time so
that symptom / disease names always line up.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import config


class DatasetError(RuntimeError):
    """Raised when a required dataset file is missing or malformed."""


# --------------------------------------------------------------------------- #
# Normalisation helpers
# --------------------------------------------------------------------------- #
def normalize_symptom(raw: str) -> str:
    """' dischromic _patches' -> 'dischromic_patches'; 'Skin Rash' -> 'skin_rash'."""
    return re.sub(r"[\s_]+", "_", str(raw).strip().lower()).strip("_")


def normalize_disease(raw: str) -> str:
    """Trim and collapse repeated whitespace; case is preserved for display."""
    return re.sub(r"\s+", " ", str(raw)).strip()


def match_key(raw: str) -> str:
    """Aggressive key used only to JOIN tables (letters and digits, lowercase).

    Makes 'foul_smell_of_urine' and 'foul_smell_ofurine' compare equal.
    """
    return re.sub(r"[^a-z0-9]", "", str(raw).lower())


def symptom_label(symptom: str) -> str:
    """Human readable label for the UI."""
    return symptom.replace("_", " ").strip()


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"Required dataset file is missing: {path.name}")
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        raise DatasetError(f"Could not read {path.name}: {exc}") from exc
    if df.empty:
        raise DatasetError(f"Dataset file is empty: {path.name}")
    return df


def load_raw_training_table(path: Path = config.DATASET_FILE) -> pd.DataFrame:
    df = _read_csv(path)
    if "Disease" not in df.columns or not any(c.startswith("Symptom_") for c in df.columns):
        raise DatasetError("dataset.csv must contain a 'Disease' column and 'Symptom_*' columns")
    return df


# --------------------------------------------------------------------------- #
# Reference data (descriptions, precautions, severity)
# --------------------------------------------------------------------------- #
@dataclass
class ReferenceData:
    descriptions: dict[str, str] = field(default_factory=dict)        # keyed by match_key(disease)
    precautions: dict[str, list[str]] = field(default_factory=dict)   # keyed by match_key(disease)
    severity: dict[str, int] = field(default_factory=dict)            # keyed by normalized symptom
    unmatched_notes: list[str] = field(default_factory=list)

    def description_for(self, disease: str) -> str | None:
        return self.descriptions.get(match_key(disease))

    def precautions_for(self, disease: str) -> list[str]:
        return list(self.precautions.get(match_key(disease), []))


def _align_keys(source_keys: set[str], target_keys: set[str], what: str, notes: list[str]) -> dict[str, str]:
    """Map keys in `source_keys` that have no exact partner onto the closest key
    in `target_keys` (difflib ratio >= 0.9). Every fuzzy join is recorded in `notes`."""
    mapping = {}
    for key in source_keys - target_keys:
        close = difflib.get_close_matches(key, list(target_keys), n=1, cutoff=0.9)
        if close:
            mapping[key] = close[0]
            notes.append(f"{what}: fuzzy-matched '{key}' -> '{close[0]}'")
    return mapping


def load_reference_data(disease_names: list[str] | None = None) -> ReferenceData:
    """Load description / precaution / severity tables.

    `disease_names` (the diseases of the training table) is used to repair
    spelling differences between files (e.g. 'hemmorhoids' vs 'hemorrhoids').
    """
    ref = ReferenceData()
    target = {match_key(d) for d in (disease_names or [])}

    desc = _read_csv(config.DESCRIPTION_FILE)
    prec = _read_csv(config.PRECAUTION_FILE)
    sev = _read_csv(config.SEVERITY_FILE)

    desc_map = {match_key(d): str(t).strip() for d, t in zip(desc["Disease"], desc["Description"])}
    prec_cols = [c for c in prec.columns if c.startswith("Precaution_")]
    prec_map = {}
    for _, row in prec.iterrows():
        items = [str(row[c]).strip() for c in prec_cols if isinstance(row[c], str) and row[c].strip()]
        prec_map[match_key(row["Disease"])] = items

    for name, table in (("description", desc_map), ("precaution", prec_map)):
        if target:
            aliases = _align_keys(set(table), target, name, ref.unmatched_notes)
            for src, dst in aliases.items():
                table[dst] = table.pop(src)
    ref.descriptions, ref.precautions = desc_map, prec_map

    for symptom, weight in zip(sev["Symptom"], sev["weight"]):
        ref.severity[match_key(symptom)] = int(weight)
    return ref


def severity_lookup(ref: ReferenceData, symptoms: list[str]) -> dict[str, int | None]:
    """Severity weight for each normalised symptom (None if absent from the file)."""
    return {s: ref.severity.get(match_key(s)) for s in symptoms}
