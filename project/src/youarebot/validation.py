"""Контракт данных: схемы Pandera и проверки целостности до обучения."""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from youarebot.data import PARTICIPANT_KEY, Dataset

MESSAGES_SCHEMA = pa.DataFrameSchema(
    {
        "dialog_id": pa.Column(str, nullable=False),
        "message": pa.Column(int, pa.Check.ge(0)),
        "participant_index": pa.Column(int, pa.Check.isin([0, 1])),
        "text": pa.Column(str, nullable=False),
    },
    unique=["dialog_id", "message"],
    strict=True,
    coerce=True,
)

LABELS_SCHEMA = pa.DataFrameSchema(
    {
        "dialog_id": pa.Column(str, nullable=False),
        "participant_index": pa.Column(int, pa.Check.isin([0, 1])),
        "is_bot": pa.Column(int, pa.Check.isin([0, 1])),
    },
    unique=PARTICIPANT_KEY,
    strict=True,
    coerce=True,
)

TEST_IDS_SCHEMA = pa.DataFrameSchema(
    {
        "dialog_id": pa.Column(str, nullable=False),
        "participant_index": pa.Column(int, pa.Check.isin([0, 1])),
        "ID": pa.Column(str, unique=True),
    },
    unique=PARTICIPANT_KEY,
    strict=True,
    coerce=True,
)

SUBMISSION_SCHEMA = pa.DataFrameSchema(
    {
        "ID": pa.Column(str, unique=True),
        "is_bot": pa.Column(float, pa.Check.in_range(0.0, 1.0)),
    },
    strict=True,
    coerce=True,
)


class DataContractError(ValueError):
    """Данные не соответствуют контракту — обучать на них нельзя."""


def _check_same_dialogs(messages: pd.DataFrame, table: pd.DataFrame, table_name: str) -> None:
    in_messages = set(messages["dialog_id"])
    in_table = set(table["dialog_id"])
    missing = in_messages - in_table
    extra = in_table - in_messages
    if missing or extra:
        raise DataContractError(
            f"{table_name}: {len(missing)} диалогов без строк и {len(extra)} строк без диалога"
        )


def validate_dataset(dataset: Dataset) -> Dataset:
    """Проверяет схемы и ссылочную целостность; возвращает данные с приведёнными типами."""
    train_messages = MESSAGES_SCHEMA.validate(dataset.train_messages)
    test_messages = MESSAGES_SCHEMA.validate(dataset.test_messages)
    train_labels = LABELS_SCHEMA.validate(dataset.train_labels)
    test_ids = TEST_IDS_SCHEMA.validate(dataset.test_ids)

    _check_same_dialogs(train_messages, train_labels, "ytrain")
    _check_same_dialogs(test_messages, test_ids, "ytest")
    overlap = set(train_messages["dialog_id"]) & set(test_messages["dialog_id"])
    if overlap:
        raise DataContractError(f"{len(overlap)} диалогов есть и в train, и в test")

    return Dataset(train_messages, train_labels, test_messages, test_ids)


def data_quality_report(dataset: Dataset) -> dict[str, float]:
    """Сводка, которая логируется в MLflow вместе с каждым запуском."""
    train, test = dataset.train_messages, dataset.test_messages
    labels = dataset.train_labels
    pairs = labels.pivot(index="dialog_id", columns="participant_index", values="is_bot")
    text_len = train["text"].str.len()
    return {
        "train_dialogs": float(train["dialog_id"].nunique()),
        "train_messages": float(len(train)),
        "test_dialogs": float(test["dialog_id"].nunique()),
        "test_messages": float(len(test)),
        "train_bot_share": float(labels["is_bot"].mean()),
        "train_dialogs_two_bots": float((pairs.sum(axis=1) == 2).sum()),
        "train_empty_messages": float((train["text"].str.strip() == "").sum()),
        "train_messages_over_1000_chars": float((text_len > 1000).sum()),
    }
