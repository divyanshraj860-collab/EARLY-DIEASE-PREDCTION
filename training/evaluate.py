"""Metrics, cross-validation, ensembling helpers and plots."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold  # noqa: E402

import config  # noqa: E402
from backend.ensemble import full_proba, soft_vote  # noqa: E402,F401


def compute_metrics(y_true, proba: np.ndarray) -> dict:
    """Metrics for one probability matrix. Predictions are argmax(proba) so that the
    reported numbers match exactly what the application would output."""
    n_classes = proba.shape[1]
    y_pred = proba.argmax(axis=1)
    try:
        auc = float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro",
                                  labels=list(range(n_classes))))
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc_ovr_macro": auc,
    }


def out_of_fold_probas(estimator, X, y, n_classes: int):
    """Stratified K-fold out-of-fold probabilities (training data only).

    Returns (oof_proba, fold_ids)."""
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof = np.zeros((len(y), n_classes))
    folds = np.zeros(len(y), dtype=int)
    for k, (tr, va) in enumerate(skf.split(X, y)):
        m = clone(estimator).fit(X[tr], y[tr])
        oof[va] = full_proba(m, X[va], n_classes)
        folds[va] = k
    return oof, folds


def fold_scores(y, proba, folds) -> dict:
    accs, f1s = [], []
    for k in np.unique(folds):
        idx = folds == k
        pred = proba[idx].argmax(axis=1)
        accs.append(accuracy_score(y[idx], pred))
        f1s.append(f1_score(y[idx], pred, average="macro", zero_division=0))
    return {
        "cv_accuracy_mean": float(np.mean(accs)), "cv_accuracy_std": float(np.std(accs)),
        "cv_f1_mean": float(np.mean(f1s)), "cv_f1_std": float(np.std(f1s)),
    }


def nested_weighted_fold_scores(oof: dict, y, folds) -> dict:
    """Unbiased validation of performance-weighted voting.

    For each held-out fold the weights are estimated ONLY from the other folds'
    out-of-fold predictions, then applied to the held-out fold. This avoids
    choosing weights on the very rows used to judge them.
    """
    accs, f1s = [], []
    for k in np.unique(folds):
        rest, held = folds != k, folds == k
        f1 = {m: f1_score(y[rest], p[rest].argmax(axis=1), average="macro", zero_division=0)
              for m, p in oof.items()}
        w = {m: v / sum(f1.values()) for m, v in f1.items()}
        pred = soft_vote({m: p[held] for m, p in oof.items()}, w).argmax(axis=1)
        accs.append(accuracy_score(y[held], pred))
        f1s.append(f1_score(y[held], pred, average="macro", zero_division=0))
    return {"cv_accuracy_mean": float(np.mean(accs)), "cv_accuracy_std": float(np.std(accs)),
            "cv_f1_mean": float(np.mean(f1s)), "cv_f1_std": float(np.std(f1s))}


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(y_true, proba, class_names, title: str, path) -> None:
    n = len(class_names)
    cm = confusion_matrix(y_true, proba.argmax(axis=1), labels=list(range(n)))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, ax=ax, cmap="GnBu", cbar=True, square=True, linewidths=0.2, linecolor="#e6eeee",
                xticklabels=class_names, yticklabels=class_names, vmin=0)
    ax.set_xlabel("Predicted disease")
    ax.set_ylabel("True disease")
    ax.set_title(f"Confusion matrix (test set) - {title}")
    ax.tick_params(axis="both", labelsize=6)
    plt.setp(ax.get_xticklabels(), rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_model_comparison(rows: list[dict], path) -> None:
    names = [r["Model"] for r in rows]
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    x = np.arange(len(names))
    w = 0.2
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colors = ["#0e6b6f", "#5aa6a0", "#b7791f", "#d9b26a"]
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * w, [r[m] for r in rows], w, label=m.replace("_", "-"), color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (test set, macro-averaged)")
    ax.set_title("Model comparison")
    ax.legend(ncol=4, loc="lower center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
