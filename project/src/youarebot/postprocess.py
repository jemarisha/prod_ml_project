"""Постобработка вероятностей с учётом структуры диалога."""

from __future__ import annotations

import numpy as np
import pandas as pd

from youarebot.data import PARTICIPANT_KEY

EPS = 1e-6


def exclusive_bot_normalize(keys: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    """Учитывает, что в диалоге не бывает двух ботов.

    Оценки участника `p` и собеседника `r` считаются независимыми, исход
    «оба бота» исключается, остальные исходы перенормируются:
    `P(бот) = p * (1 - r) / (1 - p * r)`. Если строки собеседника нет,
    вероятность остаётся как есть.
    """
    p = np.clip(np.asarray(proba, dtype=float), EPS, 1 - EPS)
    frame = keys[PARTICIPANT_KEY].reset_index(drop=True).assign(p=p)
    partner = frame.assign(participant_index=1 - frame["participant_index"]).rename(
        columns={"p": "r"}
    )
    r = frame.merge(partner, on=PARTICIPANT_KEY, how="left")["r"].to_numpy(dtype=float)
    has_partner = ~np.isnan(r)
    result = p.copy()
    result[has_partner] = (
        p[has_partner] * (1 - r[has_partner]) / (1 - p[has_partner] * r[has_partner])
    )
    return np.clip(result, EPS, 1 - EPS)
