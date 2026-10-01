"""End-to-end training pipeline.

Run from the project root:   python -m training.train
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from backend.data_loader import load_raw_training_table, load_reference_data, match_key  # noqa: E402
from training import evaluate as ev  # noqa: E402
from training import preprocess as pp  # noqa: E402

RS = config.RANDOM_STATE


def make_models() -> dict:
    """Fresh, unfitted estimators.

    * Scaling (StandardScaler) is used for the distance/margin based models (KNN, SVC)
      and sits inside a Pipeline so it is fitted on training data only.
    * Tree models and BernoulliNB (made for binary features) need no scaling.
    * SVC uses probability=True so that it can take part in soft voting.
    """
    return {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=1),
        "naive_bayes": BernoulliNB(alpha=1.0),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance")),
        "svc": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale",
                                                   probability=True, random_state=RS)),
        "decision_tree": DecisionTreeClassifier(random_state=RS),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                        max_depth=3, random_state=RS),
    }


def leakage_demo(raw: pd.DataFrame) -> dict:
    """Show how much a naive split on the raw (duplicated) file would inflate scores."""
    clean_all = pd.DataFrame({
        "disease": raw["Disease"].map(pp.normalize_disease),
        "symptoms": [tuple(sorted({pp.normalize_symptom(v) for v in r if isinstance(v, str)}))
                     for r in raw[pp.symptom_columns(raw)].to_numpy()],
    })
    tr, te = train_test_split(clean_all, test_size=config.TEST_SIZE, random_state=RS,
                              stratify=clean_all["disease"])
    seen = set(zip(tr["disease"], tr["symptoms"]))
    overlap = float(np.mean([(d, s) in seen for d, s in zip(te["disease"], te["symptoms"])]))
    vocab = pp.build_vocabulary(tr["symptoms"])
    le = LabelEncoder().fit(tr["disease"])
    rf = RandomForestClassifier(n_estimators=100, random_state=RS).fit(
        pp.vectorize(tr["symptoms"], vocab), le.transform(tr["disease"]))
    acc = float(rf.score(pp.vectorize(te["symptoms"], vocab), le.transform(te["disease"])))
    return {"naive_split_test_rows_with_exact_copy_in_train": overlap, "naive_split_rf_accuracy": acc}


def main() -> None:
    t0 = time.time()
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Inspect ------------------------------------------------------------
    report_text, facts = pp.inspect_datasets()
    (config.REPORTS_DIR / "data_inspection.txt").write_text(report_text + "\n", encoding="utf-8")
    print(report_text, "\n")

    # 2. Clean, de-duplicate, split (split happens BEFORE any fitting) ---------
    raw = load_raw_training_table()
    clean, clean_stats = pp.clean_training_table(raw)
    train_df, test_df = pp.split_train_test(clean)
    print(f"Cleaning: {clean_stats}")
    print(f"Split   : train={len(train_df)}  test={len(test_df)}  (stratified, random_state={RS})")

    # 3. Vocabulary + features: TRAIN ONLY ------------------------------------
    vocab = pp.build_vocabulary(train_df["symptoms"])
    X_train = pp.vectorize(train_df["symptoms"], vocab)
    X_test = pp.vectorize(test_df["symptoms"], vocab)
    dropped = sorted({s for lst in test_df["symptoms"] for s in lst} - set(vocab))
    print(f"Vocabulary: {len(vocab)} symptoms (test-only symptoms ignored: {dropped})")

    le = LabelEncoder().fit(train_df["disease"])
    unseen = set(test_df["disease"]) - set(le.classes_)
    assert not unseen, f"test diseases missing from training split: {unseen}"
    y_train = le.transform(train_df["disease"])
    y_test = le.transform(test_df["disease"])
    class_names = list(le.classes_)
    n_classes = len(class_names)

    # 4. Validation (stratified CV on the training split only) ------------------
    keys = config.MODEL_KEYS
    oof, folds, cv = {}, None, {}
    for key in keys:
        print(f"  CV  {key} ...", flush=True)
        oof[key], folds = ev.out_of_fold_probas(make_models()[key], X_train, y_train, n_classes)
        cv[key] = ev.fold_scores(y_train, oof[key], folds)

    f1_scores = {k: cv[k]["cv_f1_mean"] for k in keys}
    equal_w = {k: 1.0 / len(keys) for k in keys}
    perf_w = {k: f1_scores[k] / sum(f1_scores.values()) for k in keys}
    val = {
        "equal": ev.fold_scores(y_train, ev.soft_vote(oof, equal_w), folds),
        "performance_weighted": ev.nested_weighted_fold_scores(oof, y_train, folds),
    }
    gain = val["performance_weighted"]["cv_f1_mean"] - val["equal"]["cv_f1_mean"]
    selected = "performance_weighted" if gain > config.WEIGHTED_MIN_GAIN else "equal"
    print(f"Validation macro-F1  equal={val['equal']['cv_f1_mean']:.4f}  "
          f"weighted={val['performance_weighted']['cv_f1_mean']:.4f}  gain={gain:+.4f}  -> {selected}")

    # 5. Final fit on the full training split, evaluate on the untouched test split
    models, test_proba, metrics = {}, {}, {}
    for key in keys:
        print(f"  FIT {key} ...", flush=True)
        models[key] = make_models()[key].fit(X_train, y_train)
        test_proba[key] = ev.full_proba(models[key], X_test, n_classes)
        metrics[key] = {**ev.compute_metrics(y_test, test_proba[key]), **cv[key]}

    ens_proba = {
        "equal": ev.soft_vote(test_proba, equal_w),
        "performance_weighted": ev.soft_vote(test_proba, perf_w),
    }
    ens_metrics = {m: {**ev.compute_metrics(y_test, p), **val[m]} for m, p in ens_proba.items()}
    final_proba = ens_proba[selected]

    # 6. Persist artifacts --------------------------------------------------------
    for key in keys:
        joblib.dump(models[key], config.MODELS_DIR / f"{key}.pkl", compress=3)
    joblib.dump(le, config.MODELS_DIR / "label_encoder.pkl")
    joblib.dump(vocab, config.MODELS_DIR / "symptom_features.pkl")
    ensemble_config = {
        "method": "probability-based soft voting (weighted average of class probabilities)",
        "model_keys": keys,
        "selected_mode": selected,
        "weights": {"equal": equal_w, "performance_weighted": perf_w},
        "validation_macro_f1": {m: val[m]["cv_f1_mean"] for m in val},
        "selection_margin": config.WEIGHTED_MIN_GAIN,
        "random_state": RS,
        "n_classes": n_classes,
        "n_features": len(vocab),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    joblib.dump(ensemble_config, config.MODELS_DIR / "ensemble_config.pkl")

    # 7. Reports --------------------------------------------------------------------
    rows = [{"Model": config.MODEL_LABELS[k], "Accuracy": metrics[k]["accuracy"],
             "Precision": metrics[k]["precision"], "Recall": metrics[k]["recall"],
             "F1_Score": metrics[k]["f1"]} for k in keys]
    for mode, label in (("equal", "Ensemble (equal weights)"),
                        ("performance_weighted", "Ensemble (performance-weighted)")):
        m = ens_metrics[mode]
        rows.append({"Model": label, "Accuracy": m["accuracy"], "Precision": m["precision"],
                     "Recall": m["recall"], "F1_Score": m["f1"]})
    comp = pd.DataFrame(rows)
    comp.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False, float_format="%.4f")
    ev.plot_model_comparison(rows, config.REPORTS_DIR / "model_comparison.png")
    for key in keys:
        ev.plot_confusion_matrix(y_test, test_proba[key], class_names, config.MODEL_LABELS[key],
                                 config.REPORTS_DIR / f"confusion_matrix_{key}.png")
    ev.plot_confusion_matrix(y_test, final_proba, class_names, f"Ensemble ({selected})",
                             config.REPORTS_DIR / "confusion_matrix_ensemble.png")

    ref = load_reference_data(sorted(set(class_names)))
    imp = sorted(zip(vocab, models["random_forest"].feature_importances_), key=lambda t: -t[1])[:15]
    importance = [{"symptom": s, "importance": float(v), "severity": ref.severity.get(match_key(s))}
                  for s, v in imp]
    leak = leakage_demo(raw)

    payload = {
        "dataset": {**facts, **clean_stats, "train_rows": len(train_df), "test_rows": len(test_df),
                    "vocabulary_size": len(vocab), "test_only_symptoms_ignored": dropped,
                    "test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
                    "random_state": RS, "classes": class_names},
        "models": {k: {"label": config.MODEL_LABELS[k], **metrics[k]} for k in keys},
        "ensemble": {
            "selected_mode": selected,
            "weights": ensemble_config["weights"],
            "modes": {m: ens_metrics[m] for m in ens_metrics},
            "validation_macro_f1": ensemble_config["validation_macro_f1"],
        },
        "rf_symptom_importance": importance,
        "leakage_demo": leak,
        "images": ["model_comparison"] + [f"confusion_matrix_{k}" for k in keys] + ["confusion_matrix_ensemble"],
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_text_report(payload, comp)
    print(f"\nDone in {time.time() - t0:.1f}s.  Artifacts -> {config.MODELS_DIR.name}/, {config.REPORTS_DIR.name}/")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def write_text_report(p: dict, comp: pd.DataFrame) -> None:
    d, e = p["dataset"], p["ensemble"]
    L = ["EVALUATION REPORT", "=" * 70, "",
         "Data preparation",
         f"  raw rows                 : {d['raw_rows']}",
         f"  exact duplicates removed : {d['exact_duplicates_dropped']}  (before splitting, to prevent leakage)",
         f"  unique rows              : {d['unique_rows']}",
         f"  ambiguous patterns       : {d['ambiguous_symptom_patterns']} (same symptoms, different disease)",
         f"  train / test rows        : {d['train_rows']} / {d['test_rows']}  (stratified, random_state={d['random_state']})",
         f"  diseases / symptoms      : {len(d['classes'])} / {d['vocabulary_size']}",
         "", "Test-set metrics (macro-averaged over 41 diseases; predictions = argmax of probabilities)", "",
         comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "",
         "Validation (stratified {}-fold CV on the training split only)".format(d["cv_folds"])]
    for k, m in p["models"].items():
        L.append(f"  {m['label']:<18} CV accuracy {m['cv_accuracy_mean']:.4f} +/- {m['cv_accuracy_std']:.4f}   "
                 f"CV macro-F1 {m['cv_f1_mean']:.4f} +/- {m['cv_f1_std']:.4f}   test ROC-AUC (OvR) "
                 f"{m['roc_auc_ovr_macro'] if m['roc_auc_ovr_macro'] is None else round(m['roc_auc_ovr_macro'], 4)}")
    v = e["validation_macro_f1"]
    L += ["", "Ensemble weighting",
          f"  equal-weight validation macro-F1       : {v['equal']:.4f}",
          f"  performance-weighted validation macro-F1: {v['performance_weighted']:.4f}  (weights re-estimated on the other folds for each held-out fold)",
          f"  rule: weighted voting is used only if it wins by > {config.WEIGHTED_MIN_GAIN:.2f} macro-F1",
          f"  SELECTED MODE: {e['selected_mode']}", "  weights (performance-weighted): " +
          ", ".join(f"{config.MODEL_LABELS[k]} {w:.3f}" for k, w in e["weights"]["performance_weighted"].items()),
          "", "Leakage demonstration",
          f"  A naive split of the raw {d['raw_rows']} rows puts an exact copy of "
          f"{p['leakage_demo']['naive_split_test_rows_with_exact_copy_in_train']:.1%} of test rows in the training set",
          f"  and gives Random Forest accuracy {p['leakage_demo']['naive_split_rf_accuracy']:.1%}. "
          "That number is inflated by duplicates and is NOT reported as model performance.",
          "", "Caveats",
          f"  * Only {d['test_rows']} test rows (about 1-2 per disease): every metric has wide uncertainty.",
          "  * Most rows are subsets of the same disease's symptom pool, i.e. the data looks synthetic;",
          "    high scores do not imply clinical accuracy.",
          "  * Not a medical diagnosis system."]
    (config.REPORTS_DIR / "evaluation_report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
