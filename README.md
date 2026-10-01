# SymptomLens — Disease Prediction from Symptoms

An end-to-end machine learning web application that predicts the most likely disease from a set of symptoms, using a **six-model soft-voting ensemble** (Random Forest, Naive Bayes, KNN, SVC, Decision Tree, Gradient Boosting) behind a Flask REST API and an HTML/CSS/JavaScript frontend.

> **Medical disclaimer.** This application provides an AI-based preliminary prediction based on the supplied symptoms and dataset. It is not a medical diagnosis. Consult a qualified healthcare professional for medical advice. It is an educational/academic project and must not be used for clinical decisions.

## Features

- Multi-select, searchable symptom picker with chips, remove buttons and live count (all 131 symptoms are loaded from the trained model, nothing is hard-coded)
- Patient information form (age, gender, optional height, weight, blood group, conditions, allergies, medications, smoking, alcohol, family history)
- Six trained classifiers plus an ensemble, with each model's vote and confidence shown in the result
- Two ensemble modes (equal-weight and performance-weighted soft voting) compared on validation data
- Results dashboard: predicted disease, ensemble confidence, top-5 alternatives, description, precautions, severity of selected symptoms
- Model-performance page: metrics table, accuracy/F1 bars, ensemble weights, confusion matrices
- JSON REST API with input validation, sanitisation and structured errors
- Works offline (no external JS or chart libraries; Google Fonts are optional and fall back to system fonts)


## Screenshots

### Prediction interface

The main prediction screen collects patient context such as age and gender, along with selectable symptoms and optional additional information.

![SymptomLens prediction interface](screenshots/prediction-form.png)

### Prediction result

The results dashboard shows the predicted disease, ensemble confidence, model voting details, and alternative disease probabilities.

![SymptomLens prediction result](screenshots/prediction-result.png)

### Model performance

The model-performance page presents held-out test-set metrics, validation results, and comparisons across the six classifiers and the two ensemble modes.

![SymptomLens model performance](screenshots/model-performance.png)


## Architecture

```
Browser (frontend/) ──fetch──▶ Flask (app.py) ──▶ backend/validation.py
                                      │
                                      ├─▶ backend/predictor.py ──▶ models/*.pkl (loaded once at start-up)
                                      └─▶ reports/metrics.json, reports/*.png (read-only)
training/ (offline) ──▶ writes models/ and reports/
```

## Dataset information

Inspected from the uploaded files (see `reports/data_inspection.txt`):

| File | Shape | Role |
|---|---|---|
| `dataset.csv` | 4920 × 18 | `Disease` (target, 41 classes, 120 rows each) + `Symptom_1…Symptom_17` (one symptom name per cell, empty-padded) |
| `Symptom-severity.csv` | 133 × 2 | severity weight 1–7 per symptom |
| `symptom_Description.csv` | 41 × 2 | one description per disease |
| `symptom_precaution.csv` | 41 × 5 | up to four precautions per disease |

Findings that shaped the pipeline:

- **Massive duplication.** 4616 of 4920 rows are exact duplicates; only **304 unique** (disease, symptom-set) records exist (5–10 per disease).
- **Messy strings.** 130 of 131 symptom strings carry a leading space; others contain inner spaces (`dischromic _patches`, `foul_smell_of urine`, `spotting_ urination`). Disease names have trailing/double spaces (`Diabetes `, `Hypertension `, `(vertigo) Paroymsal  Positional Vertigo`).
- **Cross-file spelling differences.** `Dimorphic hemmorhoids(piles)` (dataset) vs `Dimorphic hemorrhoids(piles)` (description file); `foul_smell_of_urine` vs `foul_smell_ofurine` (severity file). Files are joined on a punctuation-insensitive key, with a logged fuzzy match (difflib ≥ 0.9) for the disease name. The severity file also contains a stray `prognosis` row, which is ignored.
- **Missing values.** The ~47k empty cells in `dataset.csv` are just padding for variable-length symptom lists. Two empty precaution cells (Allergy, Heart attack) are simply omitted.
- **No age or gender columns exist.** Age, gender and all other patient fields are therefore **not model features**. They are collected, validated and displayed in the report only.
- **The data looks synthetic.** 263 of the 304 unique rows are strict subsets of another row of the same disease; each disease behaves like a fixed symptom pool sampled at random.

## Machine-learning methodology

1. **Clean** – trim, lowercase and unify separators in symptom names; collapse whitespace in disease names; drop padding cells.
2. **De-duplicate *before* splitting** – 4920 → 304 unique records.
3. **Split** – stratified 80/20, `random_state=42` → 243 train / 61 test rows.
4. **Features** – binary symptom vector (1 = present) in a fixed, sorted order; the vocabulary (131 symptoms) is built from the **training split only** and saved to `models/symptom_features.pkl`.
5. **Target** – `LabelEncoder` fitted on training labels.
6. **Validation** – stratified 4-fold cross-validation on the training split (4 folds because the rarest disease has only 4 training rows).
7. **Final fit** on the full training split; **one-time evaluation** on the untouched test split.
8. **Persist** models, label encoder, feature list and ensemble configuration with `joblib`.

### How leakage is prevented

- Exact duplicates are removed before the split. A naive split of the raw file puts an exact copy of **100%** of test rows in the training set and yields a Random Forest accuracy of 100%; that figure is meaningless and is not reported as performance.
- The vocabulary, label encoder and scalers are fitted on training data only. Scalers live inside scikit-learn `Pipeline`s, so in cross-validation they are re-fitted on each training fold.
- Ensemble weights come from cross-validation on the training split. To compare equal vs weighted voting fairly, the weights for each held-out fold are estimated from the *other* folds only.
- The test set is used once, after every choice (hyper-parameters, ensemble mode) was fixed.

## Models used

| Model | Configuration | Preprocessing |
|---|---|---|
| Random Forest | 300 trees | none |
| Naive Bayes | `BernoulliNB` (suited to binary features), α=1 | none |
| KNN | k=5, distance-weighted | `StandardScaler` |
| SVC | RBF, C=10, `probability=True` | `StandardScaler` |
| Decision Tree | unpruned | none |
| Gradient Boosting | 100 stages, depth 3, lr 0.1 | none |

Hyper-parameters are sensible defaults, not heavily tuned (the dataset is too small for a meaningful search). A model's own prediction is defined as `argmax(predict_proba)`, so reported metrics match what the app shows.

## Ensemble methodology

The ensemble performs **soft voting: probability averaging**, not an average of predicted labels. Each model outputs a probability distribution over the 41 diseases and the distributions are combined:

```
final_probability = Σ  weight_i × probability_i        (weights sum to 1)
predicted_disease = argmax(final_probability)
```

- **Mode 1 – equal weights:** every model has weight 1/6.
- **Mode 2 – performance-weighted:** `weight_i = cv_macro_F1_i / Σ cv_macro_F1`.

Validation macro-F1: **equal 0.9577**, **performance-weighted 0.9935** (weights re-estimated on the other folds for each held-out fold). To avoid over-fitting the weighting scheme, weighted voting is only adopted if it wins by more than 0.01 macro-F1. Selected mode: **`performance_weighted`**, saved in `models/ensemble_config.pkl`.

Final weights:

| Model | Weight |
|---|---:|
| Random Forest | 0.188 |
| Naive Bayes | 0.159 |
| KNN | 0.183 |
| SVC | 0.184 |
| Decision Tree | 0.128 |
| Gradient Boosting | 0.158 |

## Model evaluation (test set, 61 records, macro-averaged)

| Model | Accuracy | Precision | Recall | F1_Score |
|---|---:|---:|---:|---:|
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9672 | 0.9553 | 0.9634 | 0.9545 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SVC | 0.9836 | 0.9919 | 0.9878 | 0.9870 |
| Decision Tree | 0.6557 | 0.5921 | 0.5976 | 0.5699 |
| Gradient Boosting | 0.8361 | 0.8022 | 0.8049 | 0.7845 |
| Ensemble (equal weights) | 0.9836 | 0.9919 | 0.9878 | 0.9870 |
| Ensemble (performance-weighted) | 0.9836 | 0.9919 | 0.9878 | 0.9870 |

Validation and ROC-AUC:

| Model | CV accuracy (4-fold) | CV macro-F1 | Test ROC-AUC (OvR) |
|---|---:|---:|---:|
| Random Forest | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 |
| Naive Bayes | 0.905 ± 0.014 | 0.846 ± 0.024 | 1.000 |
| KNN | 0.984 ± 0.012 | 0.977 ± 0.017 | 1.000 |
| SVC | 0.984 ± 0.012 | 0.982 ± 0.012 | 1.000 |
| Decision Tree | 0.741 ± 0.013 | 0.684 ± 0.029 | 0.794 |
| Gradient Boosting | 0.868 ± 0.011 | 0.841 ± 0.033 | 0.986 |

**Read these numbers with care.** The test set has only 61 records (1–2 per disease), so one error moves accuracy by about 1.6 points. Random Forest and KNN score perfectly on this split while the ensemble makes one error; with so few rows that difference is not meaningful. Decision Tree (single unpruned tree on 243 rows) and Gradient Boosting are clearly weaker, which is why performance weighting helps in validation. On this dataset the ensemble is **not** demonstrably better than Random Forest or KNN; it is used because the project requires combining all six models and because it reduces dependence on any single (weak) model. High scores reflect the synthetic structure of the dataset, not clinical accuracy.

Generated files: `reports/model_comparison.csv`, `model_comparison.png`, `confusion_matrix_<model>.png`, `confusion_matrix_ensemble.png`, `evaluation_report.txt`, `metrics.json`, `data_inspection.txt`.

## How Symptom-severity.csv is used

Severity weights (1–7) are **displayed** next to each symptom, and the total/maximum severity of the selected symptoms is shown in the report. They are **not model features**: adding a weight to a binary indicator only rescales a column that tree models are invariant to and that KNN/SVC standardise away, so there is no technical justification, and doing so could leak an arbitrary human-assigned scale into the model. Severity is also joined to the Random Forest feature-importance analysis in `reports/metrics.json`.

## Installation

```bash
cd disease_prediction_ai
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Training

```bash
python -m training.train
```

Takes about 20–30 seconds on a laptop CPU, is deterministic (`random_state=42`) and regenerates everything in `models/` and `reports/`. Trained artifacts are already included in the ZIP, so training is optional. The web app loads the saved files; it never retrains at start-up.

## Running the application

```bash
python app.py
```

Open **http://127.0.0.1:5000**. Configuration via environment variables: `HOST`, `PORT`, `FLASK_DEBUG=1`. For deployment use a production WSGI server, e.g. `gunicorn app:app`.

## Tests

```bash
python -m unittest discover -s tests -v      # or: pytest
```

28 tests cover cleaning, de-duplication, split reproducibility, vocabulary/vector generation, artifact loading and reloading, all six models, ensemble arithmetic, every API endpoint, invalid input, malformed JSON, oversized bodies, sanitisation and degraded mode when models are missing.

## API documentation

| Method | Path | Description |
|---|---|---|
| GET | `/` | Frontend |
| GET | `/api/health` | Model status (`503` if artifacts failed to load) |
| GET | `/api/symptoms` | Symptom catalogue `{name, label, severity}` |
| POST | `/api/predict` | Predict a disease |
| GET | `/api/disease/<name>` | Description and precautions for a disease |
| GET | `/api/model-performance` | Evaluation metrics (JSON) |
| GET | `/api/report-image/<name>` | Whitelisted evaluation PNGs |

Symptom names are matched after normalisation (`"Skin Rash"` → `skin_rash`). Use names from `/api/symptoms`; an unknown name returns `400` with suggestions (e.g. `fever` → `high_fever`, `mild_fever`).

### Example request

```bash
curl -X POST http://127.0.0.1:5000/api/predict -H "Content-Type: application/json" -d '{
  "age": 22, "gender": "Male",
  "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
  "additional_info": {"height": 175, "weight": 70, "blood_group": "O+"}
}'
```

`gender` ∈ Male, Female, Other, Prefer not to say. `age` 1–120. Optional `additional_info` keys: `height` (cm), `weight` (kg), `blood_group`, `smoking_status`, `alcohol_consumption`, `existing_conditions`, `allergies`, `medications`, `family_history`.

### Example response (abridged)

```json
{
  "predicted_disease": "Fungal infection",
  "ensemble_confidence": 0.731,
  "ensemble_method": "soft voting (weighted average of class probabilities)",
  "ensemble_mode": "performance_weighted",
  "model_predictions": {"random_forest": "Fungal infection", "naive_bayes": "Fungal infection", "knn": "Fungal infection",
                        "svc": "Fungal infection", "decision_tree": "Fungal infection", "gradient_boosting": "Fungal infection"},
  "model_details": {"random_forest": {"label": "Random Forest", "prediction": "...", "confidence": 0.0, "weight": 0.188}},
  "top_predictions": [{"disease": "Fungal infection", "probability": 0.731}],
  "description": "...", "precautions": ["bath twice", "..."],
  "selected_symptoms": [{"name": "itching", "label": "itching", "severity": 1}],
  "warnings": [], "patient_info": {"age": 22, "gender": "Male", "bmi": 22.9},
  "disclaimer": "This application provides an AI-based preliminary prediction ..."
}
```

Errors are `{"error": "...", "field": "age"}` with status 400 (validation), 413, 415, 503 (models missing) or 500.

## Security notes

Strict input validation and whitelisting; free text stripped of control characters and `<>`; the frontend inserts server data with `textContent` only; request bodies limited to 32 KB; static report images served only from a whitelist; no file-system paths in responses; Content-Security-Policy and other security headers; same-origin design so CORS is not enabled. `joblib` files are pickles: only load model files you trust. There is no authentication or rate limiting; add them before any public deployment.

## Project structure

```
disease_prediction_ai/
├── app.py                  Flask app factory + REST API
├── config.py               paths, constants, env-var settings
├── requirements.txt  README.md  .gitignore
├── data/                   the four source CSV files
├── backend/
│   ├── data_loader.py      normalisation + description/precaution/severity loading
│   ├── ensemble.py         soft-voting helpers (shared by training and serving)
│   ├── predictor.py        loads artifacts, runs the ensemble
│   └── validation.py       request validation / sanitisation
├── training/
│   ├── preprocess.py       inspection, cleaning, split, binary features
│   ├── evaluate.py         metrics, cross-validation, plots
│   └── train.py            full pipeline (python -m training.train)
├── models/                 6 models, ensemble_config, label_encoder, symptom_features (.pkl)
├── frontend/               index.html, style.css, script.js (single page: Predict + Model performance)
├── reports/                model_comparison.csv/.png, confusion matrices, evaluation_report.txt, metrics.json
└── tests/                  test_model.py, test_api.py
```

Differences from the suggested layout: the separate `templates/`, `static/` and `results.html` were dropped; results and performance are views of the single-page frontend, served by Flask from `frontend/`.

## Limitations

- **Synthetic, tiny dataset:** 304 unique records for 41 diseases; rows are random subsets of per-disease symptom lists. Real patients present differently.
- **Small test set** (61 rows): metrics have wide uncertainty and cannot support claims about real-world accuracy.
- **Closed world:** the system can only answer with one of the 41 diseases, even for symptoms that fit none of them. A confident output is not evidence the disease is present.
- **Few symptoms → unreliable:** every training record has at least 3 symptoms; the app warns for fewer.
- **Age, gender and history are not used** by the models (absent from the data).
- Probabilities are not calibrated; SVC (Platt scaling over 41 classes) and tree models in particular give coarse values.
- Descriptions and precautions are shown exactly as supplied, including source spelling mistakes, and have not been medically reviewed.
- Final models are trained on the training split only (the test split is held back for honest evaluation).

## Future improvements

Real clinical data with demographics; probability calibration; nested cross-validation for hyper-parameters; an "other/none of these" class; explainability (per-symptom contributions); rate limiting, authentication and HTTPS for deployment; medically reviewed content; multilingual UI.
