"""Train, evaluate and save the Random Forest and Logistic Regression models."""
import json
import time

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import build_preprocessor, clean_feature_names, split_data

MODEL_COLORS = config.MODEL_COLORS


def get_models():
    """The two classifiers. class_weight='balanced' makes the rare fraud class count more."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=config.RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=config.RANDOM_STATE),
    }


def evaluate(y_true, y_prob, threshold=config.DECISION_THRESHOLD):
    """All the scores we report for one model."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _feature_importance(pipelines):
    """Random Forest importances and Logistic Regression coefficients in one table."""
    rf, lr = pipelines["Random Forest"], pipelines["Logistic Regression"]
    names = clean_feature_names(rf.named_steps["preprocess"].get_feature_names_out())
    table = pd.DataFrame({
        "feature": names,
        "rf_importance": rf.named_steps["model"].feature_importances_,
        "lr_coefficient": lr.named_steps["model"].coef_[0],
    })
    return table.sort_values("rf_importance", ascending=False, ignore_index=True)


# ------------------------------------------------------------------ charts --
def _save(fig, name):
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confusion_matrices(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5.5 * len(results), 4.5))
    for ax, (name, res) in zip(np.atleast_1d(axes), results.items()):
        cm = res["confusion_matrix"]
        matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        sns.heatmap(matrix, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"],
                    annot_kws={"size": 13})
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}\nprecision {res['precision']:.2f} | recall {res['recall']:.2f}")
    return _save(fig, "12_confusion_matrices.png")


def plot_roc_pr_curves(y_test, probs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax1.plot(fpr, tpr, lw=2, color=MODEL_COLORS[name],
                 label=f"{name} (AUC = {roc_auc_score(y_test, prob):.3f})")
        precision, recall, _ = precision_recall_curve(y_test, prob)
        ax2.plot(recall, precision, lw=2, color=MODEL_COLORS[name],
                 label=f"{name} (AP = {average_precision_score(y_test, prob):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Random guess")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curve")
    ax2.axhline(y_test.mean(), color="k", ls="--", lw=1, label="Random guess")
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax1.legend(loc="lower right")
    ax2.legend(loc="lower left")
    return _save(fig, "13_roc_pr_curves.png")


def plot_feature_importance(importance, top_n=15):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    top_rf = importance.head(top_n).iloc[::-1]
    ax1.barh(top_rf["feature"], top_rf["rf_importance"], color=MODEL_COLORS["Random Forest"])
    ax1.set_title("Random Forest - feature importance")
    ax1.set_xlabel("Importance (mean decrease in impurity)")

    top_lr = importance.reindex(importance["lr_coefficient"].abs()
                                .sort_values(ascending=False).index).head(top_n).iloc[::-1]
    colors = [config.COLOR_FRAUD if c > 0 else config.COLOR_LEGIT for c in top_lr["lr_coefficient"]]
    ax2.barh(top_lr["feature"], top_lr["lr_coefficient"], color=colors)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_title("Logistic Regression - coefficients\n(red = raises fraud risk, blue = lowers it)")
    ax2.set_xlabel("Coefficient (on scaled features)")
    return _save(fig, "14_feature_importance.png")


def plot_model_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, (name, res) in enumerate(results.items()):
        values = [res[m] for m in metrics]
        bars = ax.bar(x + (i - 0.5) * width, values, width, label=name, color=MODEL_COLORS[name])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(x, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"])
    ax.set_ylim(0, 1.1)
    ax.set_title("Model comparison on the test set")
    ax.legend(loc="lower right")
    return _save(fig, "15_model_comparison.png")


# ---------------------------------------------------------------- training --
def train_and_evaluate(df):
    """Train both models on `df` (with add_features applied), save models and reports."""
    config.ensure_dirs()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"  Train: {len(X_train):,} rows ({y_train.sum()} fraud) | "
          f"Test: {len(X_test):,} rows ({y_test.sum()} fraud)")

    pipelines, results, probs = {}, {}, {}
    for name, model in get_models().items():
        pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        seconds = time.perf_counter() - start

        prob = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {**evaluate(y_test, prob), "train_seconds": round(seconds, 2)}
        pipelines[name], probs[name] = pipeline, prob
        joblib.dump(pipeline, config.MODEL_FILES[name], compress=3)

        print(f"\n  === {name} (trained in {seconds:.1f}s) ===")
        print(classification_report(y_test, (prob >= config.DECISION_THRESHOLD).astype(int),
                                    target_names=["Genuine", "Fraud"], digits=3))
        print(f"  ROC-AUC: {results[name]['roc_auc']:.4f} | PR-AUC: {results[name]['pr_auc']:.4f}")
        print(f"  Saved model -> {config.MODEL_FILES[name].relative_to(config.PROJECT_ROOT)}")

    # Reports used by the dashboard
    best = max(results, key=lambda n: results[n]["pr_auc"])
    with open(config.METRICS_JSON, "w", encoding="utf-8") as fh:
        json.dump({"threshold": config.DECISION_THRESHOLD, "best_model": best,
                   "test_rows": len(y_test), "models": results}, fh, indent=2)
    pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "amount": X_test["amount"].to_numpy(),
        **{f"prob_{n.lower().replace(' ', '_')}": p for n, p in probs.items()},
    }).to_csv(config.TEST_PREDICTIONS_CSV, index=False)
    importance = _feature_importance(pipelines)
    importance.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)

    charts = [plot_confusion_matrices(results), plot_roc_pr_curves(y_test, probs),
              plot_feature_importance(importance), plot_model_comparison(results)]
    for path in charts:
        print(f"  Saved chart -> {path.relative_to(config.PROJECT_ROOT)}")

    print(f"\n  Best model (by PR-AUC): {best}")
    return results
