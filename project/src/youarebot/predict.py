"""Предсказания зарегистрированной моделью: сабмит и разбор одного диалога."""

from __future__ import annotations

from pathlib import Path

import mlflow.pyfunc
import pandas as pd

from youarebot.config import DataConfig
from youarebot.data import load_dataset, read_dialogs
from youarebot.inference import build_submission


def load_registered_model(model_name: str, alias: str) -> mlflow.pyfunc.PyFuncModel:
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")


def predict_submission(
    model: mlflow.pyfunc.PyFuncModel, data: DataConfig, output: Path
) -> pd.DataFrame:
    dataset = load_dataset(data)
    submission = build_submission(dataset.test_ids, model.predict(dataset.test_messages))
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    return submission


def stream_dialog(
    model: mlflow.pyfunc.PyFuncModel, dialog_path: Path
) -> list[tuple[int, str, float]]:
    """Вероятность «бот» для автора каждой реплики по мере развития диалога.

    Модель видит только реплики до текущей включительно — так сервис будет
    работать в живом чате.
    """
    messages = read_dialogs(dialog_path)
    if messages["dialog_id"].nunique() != 1:
        raise ValueError("В файле должен быть ровно один диалог")
    results = []
    for end in range(1, len(messages) + 1):
        prefix = messages.iloc[:end]
        last = prefix.iloc[-1]
        predictions = model.predict(prefix)
        is_author = predictions["participant_index"] == last["participant_index"]
        probability = float(predictions.loc[is_author, "is_bot"].iloc[0])
        results.append((int(last["participant_index"]), str(last["text"]), probability))
    return results
