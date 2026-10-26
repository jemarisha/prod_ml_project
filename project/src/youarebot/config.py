"""Конфигурация эксперимента.

Каждый эксперимент описывается YAML-файлом, который накладывается поверх
`configs/base.yaml`. Итоговый конфиг валидируется pydantic-моделями и целиком
логируется в MLflow, поэтому любой запуск можно повторить по его параметрам.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

BASE_CONFIG_PATH = Path("configs/base.yaml")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    raw_dir: Path = Path("data/raw")
    train_dialogs: str = "train.json"
    train_labels: str = "ytrain.csv"
    test_dialogs: str = "test.json"
    test_ids: str = "ytest.csv"
    checksums_file: Path = Path("data/checksums.sha256")


class FeatureConfig(StrictModel):
    # Отдельные реплики бывают огромными (вставка текста на 65 тыс. символов),
    # для текстовых признаков их обрезаем.
    max_chars_per_message: int = Field(default=1000, gt=0)


class TfidfConfig(StrictModel):
    column: str = "text_self"
    analyzer: Literal["word", "char", "char_wb"] = "word"
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_features: int | None = 20000
    lowercase: bool = True
    sublinear_tf: bool = True


class ModelConfig(StrictModel):
    type: Literal["constant", "logreg", "catboost", "blend"]
    use_handcrafted: bool = True
    tfidf: list[TfidfConfig] = Field(default_factory=list)
    text_columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    components: list[ModelConfig] = Field(default_factory=list)
    weights: list[float] | None = None


class CVConfig(StrictModel):
    n_splits: int = Field(default=5, ge=2)


class PostprocessConfig(StrictModel):
    # В обучающих данных в одном диалоге не бывает двух ботов.
    exclusive_bot: bool = True


class TrackingConfig(StrictModel):
    experiment_name: str = "youarebot"
    registered_model_name: str = "youarebot"
    log_model: bool = True


class ExperimentConfig(StrictModel):
    run_name: str
    description: str = ""
    seed: int = 42
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    model: ModelConfig
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


class ProjectSettings(BaseModel):
    """Общие настройки из base.yaml для команд, которым не нужна модель."""

    model_config = ConfigDict(extra="ignore")

    data: DataConfig = Field(default_factory=DataConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивно накладывает `override` на `base`; списки заменяются целиком."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def parse_override(assignment: str) -> tuple[str, Any]:
    """`model.params.C=2` → ("model.params.C", 2); значение разбирается как YAML."""
    key, sep, value = assignment.partition("=")
    if not sep or not key:
        raise ValueError(f"Ожидается KEY=VALUE, получено {assignment!r}")
    return key.strip(), yaml.safe_load(value)


def _set_dotted(raw: dict[str, Any], key: str, value: Any) -> None:
    node = raw
    *parents, leaf = key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def load_config(
    path: Path,
    base_path: Path | None = BASE_CONFIG_PATH,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
    """Конфиг эксперимента поверх base.yaml; `overrides` — правки вида `model.params.C=2`.

    Каждая правка дописывается в имя запуска, чтобы варианты перебора
    различались в MLflow и в реестре.
    """
    raw: dict[str, Any] = {}
    if base_path is not None and base_path.exists():
        raw = read_yaml(base_path)
    raw = deep_merge(raw, read_yaml(path))
    parsed = [parse_override(item) for item in overrides or []]
    for key, value in parsed:
        _set_dotted(raw, key, value)
    if parsed:
        suffix = ",".join(f"{key.rsplit('.', 1)[-1]}={value}" for key, value in parsed)
        raw["run_name"] = f"{raw['run_name']}[{suffix}]"
    return ExperimentConfig.model_validate(raw)


def load_project_settings(base_path: Path = BASE_CONFIG_PATH) -> ProjectSettings:
    return ProjectSettings.model_validate(read_yaml(base_path) if base_path.exists() else {})


def flatten_params(config: ExperimentConfig) -> dict[str, str]:
    """Плоский словарь `секция.ключ -> значение` для `mlflow.log_params`."""
    flat: dict[str, str] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict) and value:
            for key, nested in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), nested)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            flat[prefix] = json.dumps(value, ensure_ascii=False)
        else:
            flat[prefix] = (
                json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value)
            )

    walk("", config.model_dump(mode="json"))
    return flat
