"""Dataset inspection, cleaning and feature construction.

Leakage notes
-------------
* ``clean_training_table`` removes exact duplicates BEFORE splitting. The raw file
  holds each (disease, symptom-set) pattern many times; splitting first would put
  identical rows in both train and test and inflate every metric.
* ``build_vocabulary`` is called on the TRAINING split only. The symptom feature
  list (and therefore the column order) never sees test rows.
* Scalers live inside sklearn Pipelines and are fitted on training folds only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from backend.data_loader import (
    load_raw_training_table,
    load_reference_data,
    match_key,
    normalize_disease,
    normalize_symptom,
    _read_csv,
)


def symptom_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("Symptom_")]


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def _frame_facts(name: str, df: pd.DataFrame) -> list[str]:
    return [
        f"[{name}]",
        f"  shape            : {df.shape[0]} rows x {df.shape[1]} columns",
        f"  columns          : {', '.join(map(str, df.columns))}",
        f"  dtypes           : {', '.join(sorted({str(t) for t in df.dtypes}))}",
        f"  missing cells    : {int(df.isna().sum().sum())}",
        f"  duplicate rows   : {int(df.duplicated().sum())}",
    ]


def inspect_datasets() -> tuple[str, dict]:
    """Return (human readable report, facts dict) describing the four source files."""
    raw = load_raw_training_table()
    desc = _read_csv(config.DESCRIPTION_FILE)
    prec = _read_csv(config.PRECAUTION_FILE)
    sev = _read_csv(config.SEVERITY_FILE)
    sc = symptom_columns(raw)

    cells = [v for v in raw[sc].to_numpy().ravel() if isinstance(v, str)]
    raw_names = sorted(set(cells))
    norm_names = sorted({normalize_symptom(v) for v in raw_names})
    messy = [v for v in raw_names if v != v.strip() or " " in v.strip() or v != v.lower()]
    diseases_raw = raw["Disease"].unique()
    messy_diseases = [d for d in diseases_raw if d != normalize_disease(d)]
    counts = raw["Disease"].map(normalize_disease).value_counts()

    disease_keys = {match_key(d) for d in raw["Disease"]}
    sev_keys = {match_key(s) for s in sev["Symptom"]}
    vocab_keys = {match_key(s) for s in norm_names}

    facts = {
        "raw_rows": int(len(raw)),
        "raw_duplicate_rows": int(raw.duplicated().sum()),
        "n_diseases": int(counts.size),
        "rows_per_disease_min": int(counts.min()),
        "rows_per_disease_max": int(counts.max()),
        "n_symptom_columns": len(sc),
        "n_raw_symptom_strings": len(raw_names),
        "n_symptoms_normalized": len(norm_names),
        "symptom_strings_needing_cleanup": len(messy),
        "disease_names_needing_cleanup": len(messy_diseases),
        "severity_rows": int(len(sev)),
        "severity_unmatched_to_vocab": sorted(sev_keys - vocab_keys),
        "vocab_unmatched_to_severity": sorted(vocab_keys - sev_keys),
    }

    lines = ["DATASET INSPECTION", "=" * 60, ""]
    for name, df in (("dataset.csv", raw), ("symptom_Description.csv", desc),
                     ("symptom_precaution.csv", prec), ("Symptom-severity.csv", sev)):
        lines += _frame_facts(name, df) + [""]
    lines += [
        "Target / features (dataset.csv)",
        "  target variable  : Disease (categorical)",
        f"  symptom columns  : {len(sc)} ({sc[0]} ... {sc[-1]}); each cell holds one symptom name or is empty",
        f"  disease labels   : {facts['n_diseases']} classes, "
        f"{facts['rows_per_disease_min']}-{facts['rows_per_disease_max']} rows each "
        f"({'perfectly balanced' if counts.min() == counts.max() else 'imbalanced'} in the raw file)",
        f"  distinct symptom strings : {len(raw_names)} raw -> {len(norm_names)} after normalisation",
        f"  strings with leading/inner spaces or capitals : {len(messy)}",
        f"  disease names with stray whitespace           : {len(messy_diseases)} {messy_diseases}",
        "",
        "Relationship between files",
        "  dataset.csv        -> training data (Disease + up to 17 symptom columns)",
        "  Symptom-severity   -> weight 1-7 per symptom (joined on a punctuation-insensitive key)",
        f"      severity rows with no symptom in dataset.csv : {facts['severity_unmatched_to_vocab']}",
        f"      dataset symptoms with no severity row        : {facts['vocab_unmatched_to_severity']}",
        "  symptom_Description / symptom_precaution -> looked up by disease name",
        f"      disease keys missing from description file  : {sorted(disease_keys - {match_key(d) for d in desc['Disease']})}",
        f"      disease keys missing from precaution file   : {sorted(disease_keys - {match_key(d) for d in prec['Disease']})}",
        "",
        "Preprocessing required: yes (whitespace/case normalisation of symptoms, whitespace",
        "cleanup of disease names, NaN padding removal, exact-duplicate removal, fuzzy join of",
        "disease names that are spelled differently across files).",
    ]
    ref = load_reference_data(sorted({normalize_disease(d) for d in diseases_raw}))
    if ref.unmatched_notes:
        lines += ["", "Spelling repairs applied when joining files:"] + [f"  {n}" for n in ref.unmatched_notes]
    return "\n".join(lines), facts


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_training_table(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalise names, drop empty padding and exact duplicates.

    Returns a frame with columns ``disease`` and ``symptoms`` (sorted tuple of
    normalised symptom names) plus a stats dict.
    """
    sc = symptom_columns(raw)
    diseases = raw["Disease"].map(normalize_disease)
    symptoms = [
        tuple(sorted({normalize_symptom(v) for v in row if isinstance(v, str) and v.strip()}))
        for row in raw[sc].to_numpy()
    ]
    df = pd.DataFrame({"disease": diseases.to_numpy(), "symptoms": symptoms})
    n_raw = len(df)
    df = df[df["symptoms"].map(len) > 0]
    n_nonempty = len(df)
    df = df.drop_duplicates().reset_index(drop=True)

    # Patterns that map to more than one disease would be genuinely ambiguous.
    per_pattern = df.groupby("symptoms")["disease"].nunique()
    stats = {
        "raw_rows": n_raw,
        "rows_without_symptoms_dropped": n_raw - n_nonempty,
        "exact_duplicates_dropped": n_nonempty - len(df),
        "unique_rows": len(df),
        "ambiguous_symptom_patterns": int((per_pattern > 1).sum()),
    }
    return df, stats


def split_train_test(df: pd.DataFrame):
    """Reproducible stratified split on the de-duplicated table."""
    return train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df["disease"]
    )


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_vocabulary(symptom_lists) -> list[str]:
    """Sorted list of every symptom seen in `symptom_lists` (pass TRAIN rows only)."""
    return sorted({s for lst in symptom_lists for s in lst})


def symptoms_to_vector(symptoms, vocabulary: list[str]) -> np.ndarray:
    """Binary vector (1 = present) in the exact order of `vocabulary`."""
    index = {s: i for i, s in enumerate(vocabulary)}
    vec = np.zeros(len(vocabulary), dtype=np.uint8)
    for s in symptoms:
        i = index.get(s)
        if i is not None:
            vec[i] = 1
    return vec


def vectorize(symptom_lists, vocabulary: list[str]) -> np.ndarray:
    index = {s: i for i, s in enumerate(vocabulary)}
    X = np.zeros((len(symptom_lists), len(vocabulary)), dtype=np.uint8)
    for r, lst in enumerate(symptom_lists):
        for s in lst:
            i = index.get(s)
            if i is not None:
                X[r, i] = 1
    return X
