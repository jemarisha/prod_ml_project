"""Метрики качества и графики для отчёта по эксперименту."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

THRESHOLD = 0.5


def classification_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    """LogLoss — метрика соревнования; остальные помогают интерпретировать модель."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    predicted = (proba >= THRESHOLD).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "logloss": float(log_loss(y_true, proba, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else float("nan"),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def save_diagnostics(y_true: np.ndarray, proba: np.ndarray, title: str, path: Path) -> Path:
    """ROC-кривая, калибровка и распределение вероятностей по классам на одной картинке."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    fpr, tpr, _ = roc_curve(y_true, proba)
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, proba):.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set(title="ROC (out-of-fold)", xlabel="FPR", ylabel="TPR")
    axes[0].legend(loc="lower right")

    prob_true, prob_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    axes[1].plot(prob_pred, prob_true, marker="o")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[1].set(title="Калибровка", xlabel="Предсказанная вероятность", ylabel="Доля ботов")

    bins = np.linspace(0, 1, 26)
    axes[2].hist(proba[y_true == 0], bins=bins, alpha=0.6, label="человек")
    axes[2].hist(proba[y_true == 1], bins=bins, alpha=0.6, label="бот")
    axes[2].set(title="Вероятность «бот» по классам", xlabel="P(бот)", ylabel="Участников")
    axes[2].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_feature_importance(importance: pd.DataFrame, path: Path, top: int = 25) -> Path:
    data = importance.sort_values("importance").tail(top)
    fig, ax = plt.subplots(figsize=(8, 0.3 * len(data) + 1))
    ax.barh(data["feature"], data["importance"])
    ax.set(title=f"Важность признаков (top-{top})", xlabel="Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
