"""MLflow-модель, которая принимает сырые сообщения диалогов.

В реестр попадает не голый классификатор, а обёртка с расчётом признаков
и постобработкой: сервису достаточно передать реплики диалога.
"""

from __future__ import annotations

import mlflow.pyfunc
import pandas as pd
from sklearn.base import ClassifierMixin

from youarebot.inference import predict_participants

# Синтетический диалог для сигнатуры и input_example модели в MLflow.
EXAMPLE_MESSAGES = pd.DataFrame(
    {
        "dialog_id": ["example"] * 4,
        "message": [0, 1, 2, 3],
        "participant_index": [0, 1, 0, 1],
        "text": ["Привет!", "привет", "ты бот?", "Нет, конечно — я человек, просто печатаю быстро."],
    }
)


class DialogBotDetector(mlflow.pyfunc.PythonModel):
    """Сообщения → вероятность «бот» для каждого участника каждого диалога."""

    def __init__(
        self, estimator: ClassifierMixin, max_chars_per_message: int, exclusive_bot: bool
    ) -> None:
        self.estimator = estimator
        self.max_chars_per_message = max_chars_per_message
        self.exclusive_bot = exclusive_bot

    def predict(self, context, model_input, params=None):
        """`model_input`: DataFrame с колонками dialog_id, message, participant_index, text."""
        return predict_participants(
            self.estimator, pd.DataFrame(model_input), self.max_chars_per_message, self.exclusive_bot
        )
