"""Чтение данных соревнования You Are Bot.

Диалоги лежат в JSON вида `{dialog_id: [{text, message, participant_index}]}`,
метки — в CSV с одной строкой на участника диалога.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from youarebot.config import DataConfig

MESSAGE_COLUMNS = ["dialog_id", "message", "participant_index", "text"]
PARTICIPANT_KEY = ["dialog_id", "participant_index"]
PARTICIPANTS_PER_DIALOG = (0, 1)


@dataclass(frozen=True)
class Dataset:
    train_messages: pd.DataFrame
    train_labels: pd.DataFrame
    test_messages: pd.DataFrame
    test_ids: pd.DataFrame


def dialogs_to_frame(dialogs: dict[str, list[dict]]) -> pd.DataFrame:
    """Словарь диалогов → таблица сообщений, отсортированная по диалогу и номеру реплики."""
    rows = [
        {
            "dialog_id": str(dialog_id),
            "message": int(message["message"]),
            "participant_index": int(message["participant_index"]),
            "text": "" if message.get("text") is None else str(message["text"]),
        }
        for dialog_id, messages in dialogs.items()
        for message in messages
    ]
    frame = pd.DataFrame(rows, columns=MESSAGE_COLUMNS)
    return frame.sort_values(["dialog_id", "message"], ignore_index=True)


def read_dialogs(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as file:
        return dialogs_to_frame(json.load(file))


def read_labels(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"dialog_id": str, "participant_index": int, "is_bot": int})


def read_test_ids(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"dialog_id": str, "participant_index": int, "ID": str})


def load_dataset(config: DataConfig) -> Dataset:
    raw = config.raw_dir
    return Dataset(
        train_messages=read_dialogs(raw / config.train_dialogs),
        train_labels=read_labels(raw / config.train_labels),
        test_messages=read_dialogs(raw / config.test_dialogs),
        test_ids=read_test_ids(raw / config.test_ids),
    )


def participant_grid(messages: pd.DataFrame) -> pd.DataFrame:
    """Все участники всех диалогов, включая тех, кто не написал ни одной реплики."""
    dialog_ids = messages["dialog_id"].drop_duplicates()
    grid = pd.MultiIndex.from_product(
        [dialog_ids, PARTICIPANTS_PER_DIALOG], names=PARTICIPANT_KEY
    )
    return grid.to_frame(index=False)


def make_submission_id(dialog_id: pd.Series, participant_index: pd.Series) -> pd.Series:
    return dialog_id.astype(str) + "_" + participant_index.astype(str)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_files(config: DataConfig) -> list[Path]:
    names = [config.train_dialogs, config.train_labels, config.test_dialogs, config.test_ids]
    return [config.raw_dir / name for name in names]


def data_checksums(config: DataConfig) -> dict[str, str]:
    return {path.name: file_sha256(path) for path in raw_files(config)}


def data_version(checksums: dict[str, str]) -> str:
    """Короткий отпечаток набора данных: меняется при изменении любого файла."""
    joined = "".join(f"{name}:{checksums[name]}" for name in sorted(checksums))
    return hashlib.sha256(joined.encode()).hexdigest()[:12]


def read_expected_checksums(path: Path) -> dict[str, str]:
    """Читает файл в формате `sha256sum`: `<hash> *<file>` или `<hash>  <file>`."""
    expected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            expected[name.lstrip("*")] = digest
    return expected
