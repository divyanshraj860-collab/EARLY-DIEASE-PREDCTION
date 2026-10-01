"""Central configuration. Values can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = Path(os.environ.get("DPA_MODELS_DIR", BASE_DIR / "models"))
REPORTS_DIR = Path(os.environ.get("DPA_REPORTS_DIR", BASE_DIR / "reports"))
FRONTEND_DIR = BASE_DIR / "frontend"

DATASET_FILE = DATA_DIR / "dataset.csv"
DESCRIPTION_FILE = DATA_DIR / "symptom_Description.csv"
PRECAUTION_FILE = DATA_DIR / "symptom_precaution.csv"
SEVERITY_FILE = DATA_DIR / "Symptom-severity.csv"

# ---- Machine learning ------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
# 4 folds because the rarest disease has only 4 training rows after de-duplication
# and splitting (StratifiedKFold needs n_splits <= smallest class size).
CV_FOLDS = 4
# Performance-weighted voting is only chosen over equal weights when it beats
# them on validation macro-F1 by more than this margin (guards against
# over-fitting a weighting scheme to noise).
WEIGHTED_MIN_GAIN = 0.01

MODEL_KEYS = [
    "random_forest",
    "naive_bayes",
    "knn",
    "svc",
    "decision_tree",
    "gradient_boosting",
]
MODEL_LABELS = {
    "random_forest": "Random Forest",
    "naive_bayes": "Naive Bayes",
    "knn": "KNN",
    "svc": "SVC",
    "decision_tree": "Decision Tree",
    "gradient_boosting": "Gradient Boosting",
}

# ---- Web application -------------------------------------------------------
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))
DEBUG = os.environ.get("FLASK_DEBUG", "0") == "1"
MAX_CONTENT_LENGTH = 32 * 1024  # request bodies larger than 32 KB are rejected
MAX_SYMPTOMS = 40
LOW_CONFIDENCE_THRESHOLD = 0.50

DISCLAIMER = (
    "This application provides an AI-based preliminary prediction based on the "
    "supplied symptoms and dataset. It is not a medical diagnosis. Consult a "
    "qualified healthcare professional for medical advice."
)
