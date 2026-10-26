"""Один эксперимент: данные → признаки → кросс-валидация → финальная модель → MLflow."""

from __future__ import annotations

import json
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import StratifiedGroupKFold

import youarebot
from youarebot.config import ExperimentConfig, flatten_params
from youarebot.data import (
    PARTICIPANT_KEY,
    data_checksums,
    data_version,
    load_dataset,
)
from youarebot.features import build_participant_features, numeric_feature_columns
from youarebot.inference import build_submission, positive_proba, predict_participants
from youarebot.metrics import (
    classification_metrics,
    save_diagnostics,
    save_feature_importance,
)
from youarebot.models import build_model, feature_importance
from youarebot.postprocess import exclusive_bot_normalize
from youarebot.pyfunc_model import EXAMPLE_MESSAGES, DialogBotDetector
from youarebot.tracking import (
    configure_mlflow,
    environment_params,
    git_revision,
    model_pip_requirements,
)
from youarebot.validation import data_quality_report, validate_dataset

SUBMISSIONS_DIR = Path("submissions")
PACKAGE_DIR = Path(youarebot.__file__).parent


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    run_name: str
    cv_logloss: float
    model_uri: str | None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def cross_validate_oof(
    model: ClassifierMixin,
    features: pd.DataFrame,
    target: np.ndarray,
    n_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold вероятности и номер фолда для каждой строки.

    Разбиение групповое по `dialog_id`: оба участника диалога попадают в один фолд,
    иначе модель видела бы собеседника на обучении, а оценка была бы завышена.
    """
    oof = np.zeros(len(target))
    folds = np.full(len(target), -1)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(features, target, groups=features["dialog_id"])
    ):
        fitted = clone(model).fit(features.iloc[train_idx], target[train_idx])
        oof[valid_idx] = positive_proba(fitted, features.iloc[valid_idx])
        folds[valid_idx] = fold
    return oof, folds


def _summarize_folds(target: np.ndarray, oof: np.ndarray, folds: np.ndarray) -> dict[str, float]:
    per_fold = pd.DataFrame(
        [classification_metrics(target[folds == k], oof[folds == k]) for k in np.unique(folds)]
    )
    summary = {f"cv_{name}": float(per_fold[name].mean()) for name in per_fold.columns}
    summary |= {f"cv_{name}_std": float(per_fold[name].std(ddof=0)) for name in per_fold.columns}
    return summary


def _prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    set_global_seed(config.seed)
    dataset = validate_dataset(load_dataset(config.data))
    checksums = data_checksums(config.data)
    exclusive = config.postprocess.exclusive_bot

    features = build_participant_features(
        dataset.train_messages,
        config.features.max_chars_per_message,
        participants=dataset.train_labels,
    )
    train = features.merge(
        dataset.train_labels, on=PARTICIPANT_KEY, how="left", validate="one_to_one"
    )
    target = train.pop("is_bot").to_numpy()
    numeric_columns = numeric_feature_columns(train)
    model = build_model(config.model, numeric_columns, config.seed)

    started = time.perf_counter()
    oof_raw, folds = cross_validate_oof(model, train, target, config.cv.n_splits, config.seed)
    cv_seconds = time.perf_counter() - started
    oof = exclusive_bot_normalize(train, oof_raw) if exclusive else oof_raw

    started = time.perf_counter()
    final_model = clone(model).fit(train, target)
    fit_seconds = time.perf_counter() - started
    train_proba = positive_proba(final_model, train)
    if exclusive:
        train_proba = exclusive_bot_normalize(train, train_proba)
    test_predictions = predict_participants(
        final_model, dataset.test_messages, config.features.max_chars_per_message, exclusive
    )
    submission = build_submission(dataset.test_ids, test_predictions)

    cv_metrics = _summarize_folds(target, oof, folds)
    metrics = {
        **cv_metrics,
        **_prefixed(classification_metrics(target, oof), "oof"),
        **_prefixed(classification_metrics(target, oof_raw), "oof_raw"),
        **_prefixed(classification_metrics(target, train_proba), "train"),
        **_prefixed(data_quality_report(dataset), "data"),
        "n_features_numeric": float(len(numeric_columns)),
        "cv_seconds": cv_seconds,
        "final_fit_seconds": fit_seconds,
        # Названия из ДЗ 3: val_* — это out-of-fold оценка кросс-валидации.
        "val_logloss": cv_metrics["cv_logloss"],
        "val_accuracy": cv_metrics["cv_accuracy"],
    }

    configure_mlflow(config.tracking.experiment_name)
    with mlflow.start_run(run_name=config.run_name, description=config.description) as run:
        mlflow.set_tags(
            {
                "model_type": config.model.type,
                "data_version": data_version(checksums),
                **git_revision(),
            }
        )
        mlflow.log_params(flatten_params(config))
        mlflow.log_params({f"data.sha256.{name}": digest for name, digest in checksums.items()})
        mlflow.log_params(environment_params())
        mlflow.log_params(
            {
                "model_type": config.model.type,
                "model_params": json.dumps(config.model.params, ensure_ascii=False),
                "data_processing_params": json.dumps(
                    {
                        "max_chars_per_message": config.features.max_chars_per_message,
                        "exclusive_bot": exclusive,
                        "use_handcrafted": config.model.use_handcrafted,
                        "n_numeric_features": len(numeric_columns),
                    }
                ),
            }
        )
        mlflow.log_metrics(metrics)
        for fold in range(config.cv.n_splits):
            fold_metrics = classification_metrics(target[folds == fold], oof[folds == fold])
            for name in ("logloss", "roc_auc", "accuracy"):
                mlflow.log_metric(f"fold_{name}", fold_metrics[name], step=fold)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "config.yaml").write_text(
                yaml.safe_dump(config.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            (tmp / "data_checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
            train[PARTICIPANT_KEY].assign(
                fold=folds, is_bot=target, proba_raw=oof_raw, proba=oof
            ).to_csv(tmp / "oof_predictions.csv", index=False)
            save_diagnostics(target, oof, f"{config.run_name}: CV logloss {metrics['cv_logloss']:.4f}",
                             tmp / "diagnostics.png")
            importance = feature_importance(final_model)
            if importance is not None:
                importance.to_csv(tmp / "feature_importance.csv", index=False)
                save_feature_importance(importance, tmp / "feature_importance.png")
            mlflow.log_artifacts(tmp_dir, artifact_path="reports")

        SUBMISSIONS_DIR.mkdir(exist_ok=True)
        submission_path = SUBMISSIONS_DIR / f"{config.run_name}.csv"
        submission.to_csv(submission_path, index=False)
        mlflow.log_artifact(str(submission_path), artifact_path="submission")

        model_uri = None
        if config.tracking.log_model:
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=DialogBotDetector(
                    final_model, config.features.max_chars_per_message, exclusive
                ),
                code_paths=[str(PACKAGE_DIR)],
                input_example=EXAMPLE_MESSAGES,
                pip_requirements=model_pip_requirements(),
            )
            model_uri = info.model_uri
            mlflow.set_tag("model_uri", model_uri)

    return ExperimentResult(run.info.run_id, config.run_name, metrics["cv_logloss"], model_uri)
