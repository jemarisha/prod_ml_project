"""Инференс: сообщения диалогов → вероятность «бот» для каждого участника.

Этот код считает предсказания на тесте при обучении и работает внутри
MLflow-модели, которую загружает сервис, поэтому путь данных везде один.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from youarebot.data import MESSAGE_COLUMNS, PARTICIPANT_KEY
from youarebot.features import build_participant_features
from youarebot.postprocess import exclusive_bot_normalize
from youarebot.validation import SUBMISSION_SCHEMA


def normalize_messages(messages: pd.DataFrame) -> pd.DataFrame:
    """Приводит входные сообщения к контракту модели и сортирует их по порядку реплик."""
    missing = sorted(set(MESSAGE_COLUMNS) - set(messages.columns))
    if missing:
        raise ValueError(f"Во входных данных нет колонок: {missing}")
    frame = messages[MESSAGE_COLUMNS].copy()
    frame["dialog_id"] = frame["dialog_id"].astype(str)
    frame["message"] = frame["message"].astype(int)
    frame["participant_index"] = frame["participant_index"].astype(int)
    frame["text"] = frame["text"].fillna("").astype(str)
    if not frame["participant_index"].isin([0, 1]).all():
        raise ValueError("participant_index должен быть 0 или 1")
    return frame.sort_values(["dialog_id", "message"], ignore_index=True)


def positive_proba(estimator: ClassifierMixin, features: pd.DataFrame) -> np.ndarray:
    bot_column = list(estimator.classes_).index(1)
    return estimator.predict_proba(features)[:, bot_column]


def predict_participants(
    estimator: ClassifierMixin,
    messages: pd.DataFrame,
    max_chars_per_message: int,
    exclusive_bot: bool,
) -> pd.DataFrame:
    """Колонки результата: `dialog_id`, `participant_index`, `is_bot`."""
    messages = normalize_messages(messages)
    features = build_participant_features(messages, max_chars_per_message)
    proba = positive_proba(estimator, features)
    if exclusive_bot:
        proba = exclusive_bot_normalize(features, proba)
    return features[PARTICIPANT_KEY].assign(is_bot=proba)


def build_submission(test_ids: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    merged = test_ids.merge(predictions, on=PARTICIPANT_KEY, how="left", validate="one_to_one")
    if merged["is_bot"].isna().any():
        raise ValueError("Для части участников теста нет предсказаний")
    return SUBMISSION_SCHEMA.validate(merged[["ID", "is_bot"]])
