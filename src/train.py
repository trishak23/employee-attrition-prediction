"""Train, compare, tune the decision threshold for and evaluate attrition models.

Usage:
    python -m src.train
"""
from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from .config import CATEGORICAL, DATA_PATH, FIGURES, ID_COLUMN, META_PATH, MODEL_PATH, NUMERIC, REPORTS, TARGET
from .features import add_features, build_preprocessor

SEED = 42
RAW_FEATURES = NUMERIC + CATEGORICAL


def make_pipeline(clf) -> Pipeline:
    """Raw columns in -> probability out (feature engineering is part of the pipeline).

    Class imbalance is handled by tuning the decision threshold rather than class weights,
    which keeps predicted probabilities well calibrated.
    """
    return Pipeline([
        ("features", FunctionTransformer(add_features)),
        ("prep", build_preprocessor()),
        ("clf", clf),
    ])


def candidates() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": make_pipeline(LogisticRegression(max_iter=3000)),
        "Random Forest": make_pipeline(RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, random_state=SEED, n_jobs=-1)),
        "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=SEED)),
    }


def best_f1_threshold(y_true, proba) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    return float(thresholds[int(np.argmax(f1))])


def train(df: pd.DataFrame, save: bool = True) -> dict:
    X = df[RAW_FEATURES]
    y = (df[TARGET] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_auc = {}
    for name, pipe in candidates().items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        cv_auc[name] = float(scores.mean())
        print(f"{name:<20} CV ROC-AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

    best_name = max(cv_auc, key=cv_auc.get)
    model = candidates()[best_name]
    # threshold chosen on out-of-fold predictions so the test set stays untouched
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    threshold = best_f1_threshold(y_train, oof)

    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "best_model": best_name,
        "cv_roc_auc": cv_auc,
        "threshold": round(threshold, 3),
        "test_roc_auc": round(float(roc_auc_score(y_test, proba)), 3),
        "test_pr_auc": round(float(average_precision_score(y_test, proba)), 3),
        "test_precision": round(float(precision_score(y_test, pred)), 3),
        "test_recall": round(float(recall_score(y_test, pred)), 3),
        "test_f1": round(float(f1_score(y_test, pred)), 3),
        "attrition_rate_test": round(float(y_test.mean()), 3),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    print(f"\nBest model: {best_name} | threshold {metrics['threshold']}")
    for k in ("test_roc_auc", "test_pr_auc", "test_precision", "test_recall", "test_f1"):
        print(f"  {k:<15} {metrics[k]}")

    if save:
        REPORTS.mkdir(exist_ok=True)
        FIGURES.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        META_PATH.write_text(json.dumps(metrics, indent=2))
        (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name=best_name)
        axes[0].plot([0, 1], [0, 1], "k--", lw=1)
        axes[0].set_title("ROC curve")
        ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=["Stay", "Leave"], cmap="Blues", ax=axes[1], colorbar=False)
        axes[1].set_title(f"Confusion matrix (threshold {threshold:.2f})")
        fig.tight_layout()
        fig.savefig(FIGURES / "roc_confusion.png", dpi=150)
        plt.close(fig)

        imp = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED, n_jobs=-1)
        top = pd.Series(imp.importances_mean, index=RAW_FEATURES).sort_values().tail(12)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(top.index, top.values, color="#6366f1")
        ax.set(title="Top drivers of attrition (permutation importance)", xlabel="Drop in ROC-AUC when shuffled")
        fig.tight_layout()
        fig.savefig(FIGURES / "feature_importance.png", dpi=150)
        plt.close(fig)
        top.sort_values(ascending=False).round(4).to_csv(REPORTS / "feature_importance.csv", header=["importance"])
        print("\nTop drivers:\n" + top.sort_values(ascending=False).head(8).round(4).to_string())
        print(f"\nModel saved to {MODEL_PATH}")
    return metrics


def main() -> None:
    if not DATA_PATH.exists():
        raise SystemExit("data/employees.csv not found. Run `python -m src.generate_data` first.")
    train(pd.read_csv(DATA_PATH))


if __name__ == "__main__":
    main()
