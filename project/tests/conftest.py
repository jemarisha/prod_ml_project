"""Синтетические диалоги: тесты не зависят от данных соревнования."""

from pathlib import Path

import pandas as pd
import pytest

from youarebot.data import dialogs_to_frame

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HUMAN_LINES = ["привет", "ты бот?", "ахаха", "норм", "откуда ты", "ок"]
BOT_LINES = ["Привет, как дела?", "Я человек, конечно.", "Интересно — расскажи подробнее."]


def make_dialogs(n_dialogs: int = 24, prefix: str = "dialog") -> tuple[pd.DataFrame, pd.DataFrame]:
    """В чётных диалогах участник 1 — бот: пишет «правильно» и повторяет собеседника."""
    dialogs, labels = {}, []
    for i in range(n_dialogs):
        dialog_id = f"{prefix}-{i:03d}"
        bot = 1 if i % 2 == 0 else None
        messages: list[dict] = []
        for k in range(6):
            participant = k % 2
            if participant == bot:
                text = messages[-1]["text"] if k % 4 == 1 else BOT_LINES[(i + k) % len(BOT_LINES)]
            else:
                text = HUMAN_LINES[(i + k) % len(HUMAN_LINES)]
            messages.append({"text": text, "message": k, "participant_index": participant})
        dialogs[dialog_id] = messages
        labels += [
            {"dialog_id": dialog_id, "participant_index": p, "is_bot": int(p == bot)} for p in (0, 1)
        ]
    return dialogs_to_frame(dialogs), pd.DataFrame(labels)


@pytest.fixture
def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_dialogs()
