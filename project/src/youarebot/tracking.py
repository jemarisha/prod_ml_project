"""Подключение к MLflow и метаданные запуска: версии кода, данных и окружения."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
from pathlib import Path

import mlflow

from youarebot.data import file_sha256

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
ENV_PACKAGES = ["catboost", "mlflow", "numpy", "pandas", "pandera", "pydantic", "scikit-learn"]
MODEL_REQUIREMENTS = [
    "catboost",
    "cloudpickle",
    "mlflow",
    "numpy",
    "pandas",
    "pandera",
    "pydantic",
    "PyYAML",
    "scikit-learn",
]


def configure_mlflow(experiment_name: str) -> str:
    """URI берётся из MLFLOW_TRACKING_URI, по умолчанию — локальная SQLite-база проекта."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri


def git_revision() -> dict[str, str]:
    # git ищется в PATH намеренно: путь к нему разный на Windows, Linux и в CI.
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unknown", "git_dirty": "unknown"}
    return {"git_commit": commit, "git_dirty": str(bool(status.strip()))}


def environment_params() -> dict[str, str]:
    params = {"env.python": platform.python_version(), "env.platform": platform.platform()}
    for package in ENV_PACKAGES:
        params[f"env.{package}"] = importlib.metadata.version(package)
    lock = Path("uv.lock")
    if lock.exists():
        params["env.uv_lock_sha256"] = file_sha256(lock)
    return params


def model_pip_requirements() -> list[str]:
    """Точные версии пакетов, нужных для загрузки модели вне проекта."""
    return [f"{package}=={importlib.metadata.version(package)}" for package in MODEL_REQUIREMENTS]
