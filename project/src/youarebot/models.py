"""Фабрика моделей по конфигу.

Все модели — sklearn-совместимые классификаторы над таблицей признаков
участника, поэтому кросс-валидация, логирование и инференс для них одинаковы.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from youarebot.config import ModelConfig, TfidfConfig

CATBOOST_DEFAULTS = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}


class CatBoostOnColumns(ClassifierMixin, BaseEstimator):
    """CatBoost на выбранных колонках таблицы признаков.

    CatBoostClassifier создаётся внутри `fit`: сам он копирует список
    `text_features` в конструкторе, и `sklearn.clone` на кросс-валидации падает.
    """

    def __init__(
        self,
        columns: Sequence[str] = (),
        text_columns: Sequence[str] = (),
        params: dict | None = None,
        seed: int = 0,
    ):
        self.columns = columns
        self.text_columns = text_columns
        self.params = params
        self.seed = seed

    def _select(self, X: pd.DataFrame) -> pd.DataFrame:
        selected = X.loc[:, list(self.columns)].copy()
        for column in self.text_columns:
            selected[column] = selected[column].astype(object)
        return selected

    def fit(self, X: pd.DataFrame, y) -> CatBoostOnColumns:
        self.model_ = CatBoostClassifier(
            random_seed=self.seed,
            text_features=list(self.text_columns) or None,
            **{**CATBOOST_DEFAULTS, **(self.params or {})},
        )
        self.model_.fit(self._select(X), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(self._select(X))

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(self._select(X))


def _tfidf(config: TfidfConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=tuple(config.ngram_range),
        min_df=config.min_df,
        max_features=config.max_features,
        lowercase=config.lowercase,
        sublinear_tf=config.sublinear_tf,
    )


def _build_logreg(config: ModelConfig, numeric_columns: list[str], seed: int) -> Pipeline:
    transformers: list[tuple] = [
        (f"tfidf_{i}_{tfidf.column}_{tfidf.analyzer}", _tfidf(tfidf), tfidf.column)
        for i, tfidf in enumerate(config.tfidf)
    ]
    if config.use_handcrafted:
        transformers.append(("handcrafted", StandardScaler(), numeric_columns))
    if not transformers:
        raise ValueError("logreg: не заданы ни TF-IDF, ни ручные признаки")
    params = {"max_iter": 3000, **config.params}
    return Pipeline(
        [
            ("features", ColumnTransformer(transformers, sparse_threshold=1.0)),
            ("clf", LogisticRegression(random_state=seed, **params)),
        ]
    )


def _build_catboost(config: ModelConfig, numeric_columns: list[str], seed: int) -> CatBoostOnColumns:
    columns = (list(numeric_columns) if config.use_handcrafted else []) + list(config.text_columns)
    if not columns:
        raise ValueError("catboost: не заданы ни ручные признаки, ни текстовые колонки")
    return CatBoostOnColumns(columns, list(config.text_columns), dict(config.params), seed)


def build_model(config: ModelConfig, numeric_columns: list[str], seed: int) -> ClassifierMixin:
    if config.type == "constant":
        return DummyClassifier(strategy="prior")
    if config.type == "logreg":
        return _build_logreg(config, numeric_columns, seed)
    if config.type == "catboost":
        return _build_catboost(config, numeric_columns, seed)
    if config.type == "blend":
        if not config.components:
            raise ValueError("blend: не заданы компоненты")
        estimators = [
            (f"{i}_{component.type}", build_model(component, numeric_columns, seed))
            for i, component in enumerate(config.components)
        ]
        return VotingClassifier(estimators, voting="soft", weights=config.weights)
    raise ValueError(f"Неизвестный тип модели: {config.type}")


def feature_importance(model: BaseEstimator) -> pd.DataFrame | None:
    """Важность признаков для CatBoost; для остальных моделей — None."""
    if not isinstance(model, CatBoostOnColumns):
        return None
    return model.model_.get_feature_importance(prettified=True).rename(
        columns={"Feature Id": "feature", "Importances": "importance"}
    )
