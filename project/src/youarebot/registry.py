"""Model Registry: две лучшие конфигурации получают алиасы champion и challenger."""

from __future__ import annotations

from dataclasses import dataclass

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run

ALIASES = ("champion", "challenger")


@dataclass(frozen=True)
class Promotion:
    alias: str
    version: str
    run_name: str
    metric: float


def _latest_run_per_name(runs: list[Run], metric: str) -> list[Run]:
    """Повторный запуск того же конфига заменяет предыдущий, а не занимает второе место."""
    latest: dict[str, Run] = {}
    for run in sorted(runs, key=lambda item: item.info.start_time, reverse=True):
        if metric in run.data.metrics and "model_uri" in run.data.tags:
            latest.setdefault(run.info.run_name, run)
    return list(latest.values())


def _ensure_registered(client: MlflowClient, model_name: str, run: Run) -> str:
    existing = client.search_model_versions(
        f"name='{model_name}' and run_id='{run.info.run_id}'"
    )
    if existing:
        return existing[0].version
    version = mlflow.register_model(run.data.tags["model_uri"], model_name).version
    client.set_model_version_tag(model_name, version, "run_name", run.info.run_name)
    client.set_model_version_tag(model_name, version, "data_version", run.data.tags.get("data_version", ""))
    return version


def promote_best_runs(experiment_name: str, model_name: str, metric: str = "cv_logloss") -> list[Promotion]:
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Эксперимент {experiment_name!r} не найден — сначала запустите обучение")
    runs = client.search_runs(
        [experiment.experiment_id], filter_string="attributes.status = 'FINISHED'", max_results=5000
    )
    ranked = sorted(_latest_run_per_name(runs, metric), key=lambda run: run.data.metrics[metric])
    if len(ranked) < len(ALIASES):
        raise RuntimeError(f"Нужно минимум {len(ALIASES)} завершённых запуска с моделью")

    promotions = []
    for alias, run in zip(ALIASES, ranked, strict=False):
        version = _ensure_registered(client, model_name, run)
        client.set_registered_model_alias(model_name, alias, version)
        promotions.append(Promotion(alias, version, run.info.run_name, run.data.metrics[metric]))
    return promotions
