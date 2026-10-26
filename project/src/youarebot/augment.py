"""Аугментация обучающей выборки началами диалогов.

Начало диалога (первые k реплик) получает метки целого диалога. Так модель
видит больше коротких диалогов — на них она ошибается чаще всего — и учится
оценивать незаконченный разговор, как в живом чате.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from youarebot.data import PARTICIPANT_KEY

PREFIX_SEPARATOR = "#prefix"


@dataclass(frozen=True)
class Prefixes:
    messages: pd.DataFrame
    labels: pd.DataFrame


def make_prefixes(
    messages: pd.DataFrame,
    labels: pd.DataFrame,
    per_dialog: int,
    seed: int,
    min_length: int = 2,
) -> Prefixes:
    """До `per_dialog` случайных начал длины от `min_length` до n-1 для каждого диалога.

    Идентификатор начала — `<dialog_id>#prefix<k>`; исходный диалог восстанавливается
    функцией `source_dialog`, чтобы начала попадали в тот же фолд кросс-валидации.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for dialog_id, dialog in messages.groupby("dialog_id", sort=False):
        candidates = np.arange(min_length, len(dialog))
        if len(candidates) == 0:
            continue
        lengths = rng.choice(candidates, size=min(per_dialog, len(candidates)), replace=False)
        for length in sorted(lengths):
            parts.append(
                dialog.iloc[:length].assign(dialog_id=f"{dialog_id}{PREFIX_SEPARATOR}{length}")
            )
    if not parts:
        return Prefixes(messages.iloc[0:0], labels.iloc[0:0])
    prefix_messages = pd.concat(parts, ignore_index=True)
    sources = prefix_messages[["dialog_id"]].drop_duplicates()
    sources["source"] = source_dialog(sources["dialog_id"])
    prefix_labels = sources.merge(labels.rename(columns={"dialog_id": "source"}), on="source").drop(
        columns="source"
    )
    return Prefixes(prefix_messages, prefix_labels[[*PARTICIPANT_KEY, "is_bot"]])


def source_dialog(dialog_id: pd.Series) -> pd.Series:
    return dialog_id.str.split(PREFIX_SEPARATOR, n=1, regex=False).str[0]
