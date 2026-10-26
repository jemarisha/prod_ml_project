"""Фабрика моделей по конфигу.

Все модели — sklearn-совместимые классификаторы над таблицей признаков
участника, поэтому кросс-валидация, логирование и инференс для них одинаковы.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from youarebot.augment import PREFIX_SEPARATOR, source_dialog
from youarebot.config import ModelConfig, TfidfConfig

CATBOOST_DEFAULTS = {
    "iterations": 600,
    "learning_rate": 0.05,
    "depth": 6,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}


class WeightedPipeline(Pipeline):
    """Pipeline, принимающий `sample_weight` напрямую, как обычный классификатор.

    Без этого VotingClassifier не может передать веса примеров логрегрессии
    внутри пайплайна: штатный Pipeline ждёт параметр вида `clf__sample_weight`.
    """

    def fit(self, X, y=None, sample_weight=None, **params):
        if sample_weight is not None:
            params[f"{self.steps[-1][0]}__sample_weight"] = sample_weight
        return super().fit(X, y, **params)


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

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> CatBoostOnColumns:
        self.model_ = CatBoostClassifier(
            random_seed=self.seed,
            text_features=list(self.text_columns) or None,
            **{**CATBOOST_DEFAULTS, **(self.params or {})},
        )
        self.model_.fit(self._select(X), y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame):
        return self.model_.predict_proba(self._select(X))

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(self._select(X))


def _positive_proba(estimator: ClassifierMixin, X: pd.DataFrame) -> np.ndarray:
    return estimator.predict_proba(X)[:, list(estimator.classes_).index(1)]


def _logit(proba: np.ndarray) -> np.ndarray:
    proba = np.clip(proba, 1e-6, 1 - 1e-6)
    return np.log(proba / (1 - proba))


class LogitStack(ClassifierMixin, BaseEstimator):
    """Стекинг: логистическая регрессия поверх логитов базовых моделей.

    Метамодель учится на out-of-fold предсказаниях внутренней кросс-валидации,
    сгруппированной по диалогам. Начала диалогов (строки с `#prefix` в
    `dialog_id`) получают только базовые модели с `use_prefixes`, причём лишь
    из обучающих диалогов фолда; метамодель видит только целые диалоги.
    Усреднение вероятностей делает прогноз недоуверенным, а взвешенная сумма
    логитов со свободным масштабом этого недостатка лишена.
    """

    def __init__(
        self,
        estimators: Sequence[tuple[str, ClassifierMixin]] = (),
        use_prefixes: Sequence[bool] = (),
        n_splits: int = 5,
        C: float = 1.0,
        seed: int = 0,
    ):
        self.estimators = estimators
        self.use_prefixes = use_prefixes
        self.n_splits = n_splits
        self.C = C
        self.seed = seed

    def fit(self, X: pd.DataFrame, y, sample_weight=None) -> LogitStack:
        y = np.asarray(y)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        dialog_ids = X["dialog_id"].astype(str)
        is_prefix = dialog_ids.str.contains(PREFIX_SEPARATOR, regex=False).to_numpy()
        source = source_dialog(dialog_ids).to_numpy()
        whole = np.flatnonzero(~is_prefix)

        def fit_component(estimator, use_prefixes, base_rows, dialogs):
            rows = base_rows
            if use_prefixes:
                prefix_rows = np.flatnonzero(is_prefix & np.isin(source, dialogs))
                rows = np.concatenate([base_rows, prefix_rows])
            return clone(estimator).fit(X.iloc[rows], y[rows], sample_weight=weight[rows])

        components = list(zip(self.estimators, self.use_prefixes, strict=True))
        oof = np.zeros((len(whole), len(components)))
        splitter = StratifiedGroupKFold(self.n_splits, shuffle=True, random_state=self.seed)
        for train_pos, valid_pos in splitter.split(whole, y[whole], groups=source[whole]):
            dialogs = np.unique(source[whole[train_pos]])
            valid = X.iloc[whole[valid_pos]]
            for j, ((_, estimator), use_prefixes) in enumerate(components):
                fitted = fit_component(estimator, use_prefixes, whole[train_pos], dialogs)
                oof[valid_pos, j] = _positive_proba(fitted, valid)

        self.meta_ = LogisticRegression(C=self.C, max_iter=1000).fit(_logit(oof), y[whole])
        all_dialogs = np.unique(source[whole])
        self.estimators_ = [
            fit_component(estimator, use_prefixes, whole, all_dialogs)
            for (_, estimator), use_prefixes in components
        ]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logits = np.column_stack([_logit(_positive_proba(e, X)) for e in self.estimators_])
        return self.meta_.predict_proba(logits)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _tfidf(config: TfidfConfig) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=tuple(config.ngram_range),
        min_df=config.min_df,
        max_features=config.max_features,
        lowercase=config.lowercase,
        sublinear_tf=config.sublinear_tf,
    )


def _build_logreg(config: ModelConfig, numeric_columns: list[str], seed: int) -> WeightedPipeline:
    transformers: list[tuple] = [
        (f"tfidf_{i}_{tfidf.column}_{tfidf.analyzer}", _tfidf(tfidf), tfidf.column)
        for i, tfidf in enumerate(config.tfidf)
    ]
    if config.use_handcrafted:
        transformers.append(("handcrafted", StandardScaler(), numeric_columns))
    if not transformers:
        raise ValueError("logreg: не заданы ни TF-IDF, ни ручные признаки")
    params = {"max_iter": 3000, **config.params}
    return WeightedPipeline(
        [
            ("features", ColumnTransformer(transformers, sparse_threshold=1.0)),
            ("clf", LogisticRegression(random_state=seed, **params)),
        ]
    )


def _build_catboost(
    config: ModelConfig, numeric_columns: list[str], seed: int
) -> CatBoostOnColumns:
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
    if config.type == "stack":
        if not config.components:
            raise ValueError("stack: не заданы компоненты")
        estimators = [
            (f"{i}_{component.type}", build_model(component, numeric_columns, seed))
            for i, component in enumerate(config.components)
        ]
        use_prefixes = [component.use_prefixes for component in config.components]
        return LogitStack(estimators, use_prefixes, seed=seed, **config.params)
    raise ValueError(f"Неизвестный тип модели: {config.type}")


def feature_importance(model: BaseEstimator) -> pd.DataFrame | None:
    """Важность признаков для CatBoost; для остальных моделей — None."""
    if not isinstance(model, CatBoostOnColumns):
        return None
    return model.model_.get_feature_importance(prettified=True).rename(
        columns={"Feature Id": "feature", "Importances": "importance"}
    )
