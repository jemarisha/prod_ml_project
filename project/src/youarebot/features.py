"""Признаки участника диалога.

Одна строка на участника: его реплики, реплики собеседника и структура диалога.
`build_participant_features` вызывается и при обучении, и внутри MLflow-модели
на инференсе, поэтому признаки в обучении и в сервисе считаются одним кодом.

Регулярные выражения применяются через модуль `re` к обычным строкам Python,
а не через `.str`-методы pandas: у строк на pyarrow `\\w` понимает только ASCII,
и кириллица считалась бы «не буквами».
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from youarebot.data import PARTICIPANT_KEY, participant_grid

TEXT_COLUMNS = ["text_self", "text_other"]
NON_FEATURE_COLUMNS = ["dialog_id", *TEXT_COLUMNS]

# Доля реплик участника, на которых срабатывает шаблон, становится признаком `<имя>_share`.
MESSAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern)
    for name, pattern in {
        "starts_upper": r"^\W*[A-ZА-ЯЁ]",
        "ends_dot": r"\.\s*$",
        "ends_question": r"\?\s*$",
        "ends_exclaim": r"!\s*$",
        "ends_bracket": r"[()]\s*$",
        "has_comma": r",",
        "has_dash": r"[—–]",
        "has_ellipsis": r"\.\.\.|…",
        "has_newline": r"\n",
        "has_cyrillic": r"[а-яА-ЯёЁ]",
        "has_latin": r"[a-zA-Z]",
        "has_other_script": r"[^\W\d_a-zA-Zа-яА-ЯёЁ]",
        "has_digit": r"\d",
        "has_emoji": "[\U0001f300-\U0001faff☀-➿]",
        "mentions_bot": r"(?i)бот|bot",
        "has_laugh": r"(?i)(?:ха){2,}|(?:ah){2,}|lol|лол|\){2,}",
    }.items()
}
INTERACTION_FLAGS = [
    "is_echo",
    "echo_any",
    "self_repeat",
    "follows_self",
    "gets_reply",
    "is_doubled",
    "is_empty",
    "is_all_lower",
]
GREETING = re.compile(r"(?i)^\W*(привет|здравств|добр|хай|ку\b|hi\b|hello|hey|салам|йо\b)")
SHARE_FLAGS = [*MESSAGE_PATTERNS, *INTERACTION_FLAGS]

# Для этих признаков дополнительно считается разница «участник минус собеседник».
DIFF_FEATURES = [
    "n_msgs",
    "log_len_mean",
    "is_echo_share",
    "has_comma_share",
    "starts_upper_share",
    "ends_dot_share",
    "has_dash_share",
    "punct_ratio_mean",
]

LETTER = re.compile(r"[^\W\d_]")
UPPER_LETTER = re.compile(r"[A-ZА-ЯЁ]")
PUNCTUATION = re.compile(r"[^\w\s]")
BLOCK_SEPARATOR = re.compile(r"\n\s*\n")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def char_trigrams(text: str) -> set[str]:
    norm = normalize_text(text)
    if len(norm) < 3:
        return {norm} if norm else set()
    return {norm[i : i + 3] for i in range(len(norm) - 2)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def is_doubled(text: str) -> bool:
    """Реплика из одинаковых блоков через пустую строку: сбой бота вида «норм\\n\\nнорм»."""
    blocks = [normalize_text(block) for block in BLOCK_SEPARATOR.split(text) if block.strip()]
    return len(blocks) >= 2 and len(set(blocks)) == 1


def _interaction_features(messages: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает каждую реплику с предыдущими: эхо собеседника и самоповторы.

    Ожидает сообщения, отсортированные по диалогу и номеру реплики.
    """
    records = []
    current_dialog = None
    seen: dict[int, set[str]] = {}
    prev_text, prev_participant = "", None
    for dialog_id, participant, text in zip(
        messages["dialog_id"], messages["participant_index"], messages["text"], strict=True
    ):
        participant = int(participant)
        if dialog_id != current_dialog:
            current_dialog = dialog_id
            seen = {0: set(), 1: set()}
            prev_text, prev_participant = "", None
        norm = normalize_text(text)
        other = 1 - participant
        replies_to_other = prev_participant == other
        records.append(
            {
                "is_echo": replies_to_other and norm != "" and norm == normalize_text(prev_text),
                "echo_any": norm != "" and norm in seen[other],
                "self_repeat": norm != "" and norm in seen[participant],
                "follows_self": prev_participant == participant,
                "prev_similarity": (
                    jaccard(char_trigrams(text), char_trigrams(prev_text)) if replies_to_other else 0.0
                ),
            }
        )
        seen[participant].add(norm)
        prev_text, prev_participant = text, participant
    return pd.DataFrame.from_records(records, index=messages.index)


def _message_features(messages: pd.DataFrame) -> pd.DataFrame:
    texts = messages["text"].astype(object)
    features = pd.DataFrame(index=messages.index)
    features["len_chars"] = texts.map(len)
    features["n_words"] = texts.map(lambda text: len(text.split()))
    letters = texts.map(lambda text: len(LETTER.findall(text))).clip(lower=1)
    features["upper_ratio"] = texts.map(lambda text: len(UPPER_LETTER.findall(text))) / letters
    features["punct_ratio"] = texts.map(
        lambda text: len(PUNCTUATION.findall(text))
    ) / features["len_chars"].clip(lower=1)
    for name, pattern in MESSAGE_PATTERNS.items():
        features[name] = texts.map(lambda text, pattern=pattern: pattern.search(text) is not None)
    features["is_doubled"] = texts.map(is_doubled)
    features["is_empty"] = texts.map(lambda text: text.strip() == "")
    features["is_all_lower"] = texts.map(lambda text: bool(LETTER.search(text)) and text == text.lower())
    return features.join(_interaction_features(messages)).join(_turn_features(messages, features))


def _turn_features(messages: pd.DataFrame, per_message: pd.DataFrame) -> pd.DataFrame:
    """Ритм диалога: бот отвечает на каждую реплику и подстраивает длину ответа."""
    by_dialog = messages.groupby("dialog_id", sort=False)
    participant = messages["participant_index"]
    prev_participant = by_dialog["participant_index"].shift(1)
    next_participant = by_dialog["participant_index"].shift(-1)
    log_len = np.log1p(per_message["len_chars"])
    prev_log_len = log_len.groupby(messages["dialog_id"], sort=False).shift(1)
    replies = prev_participant.notna() & (prev_participant != participant)
    new_run = (participant != prev_participant) | prev_participant.isna()
    run_id = new_run.cumsum()
    return pd.DataFrame(
        {
            "gets_reply": next_participant.notna() & (next_participant != participant),
            "reply_len_ratio": (log_len - prev_log_len).where(replies),
            "run_length": run_id.groupby(run_id).cumcount() + 1,
        },
        index=messages.index,
    )


def _aggregate_participants(messages: pd.DataFrame, per_message: pd.DataFrame) -> pd.DataFrame:
    frame = pd.concat([messages[PARTICIPANT_KEY], per_message], axis=1)
    grouped = frame.groupby(PARTICIPANT_KEY, sort=False)
    aggregated = grouped[SHARE_FLAGS].mean().astype(float).add_suffix("_share")
    aggregated["n_msgs"] = grouped.size()
    aggregated["log_len_mean"] = np.log1p(grouped["len_chars"].mean())
    aggregated["log_len_max"] = np.log1p(grouped["len_chars"].max())
    aggregated["log_len_std"] = np.log1p(grouped["len_chars"].std().fillna(0.0))
    aggregated["words_mean"] = grouped["n_words"].mean()
    aggregated["upper_ratio_mean"] = grouped["upper_ratio"].mean()
    aggregated["punct_ratio_mean"] = grouped["punct_ratio"].mean()
    aggregated["prev_similarity_mean"] = grouped["prev_similarity"].mean()
    aggregated["prev_similarity_max"] = grouped["prev_similarity"].max()
    aggregated["reply_len_ratio_mean"] = grouped["reply_len_ratio"].mean().fillna(0.0)
    aggregated["max_run"] = grouped["run_length"].max()
    first_text = messages.groupby(PARTICIPANT_KEY, sort=False)["text"].first().astype(object)
    aggregated["first_log_len"] = np.log1p(first_text.map(len))
    aggregated["first_is_greeting"] = first_text.map(lambda text: GREETING.search(text) is not None)
    aggregated["first_is_greeting"] = aggregated["first_is_greeting"].astype(float)
    return aggregated.reset_index()


def _participant_texts(messages: pd.DataFrame, max_chars_per_message: int) -> pd.DataFrame:
    clipped = messages["text"].astype(object).map(lambda text: text[:max_chars_per_message])
    texts = clipped.groupby(
        [messages["dialog_id"], messages["participant_index"]], sort=False
    ).agg("\n".join)
    return texts.rename("text_self").reset_index()


def _swap_participant(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(participant_index=1 - frame["participant_index"])


def build_participant_features(
    messages: pd.DataFrame,
    max_chars_per_message: int = 1000,
    participants: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Таблица признаков: одна строка на участника в порядке `participants`.

    Если `participants` не передан, берутся оба участника каждого диалога.
    Участник без реплик получает нулевые признаки и пустой текст.
    """
    messages = messages.sort_values(["dialog_id", "message"], ignore_index=True)
    if participants is None:
        participants = participant_grid(messages)
    frame = participants[PARTICIPANT_KEY].reset_index(drop=True)

    own = _aggregate_participants(messages, _message_features(messages))
    numeric = [column for column in own.columns if column not in PARTICIPANT_KEY]
    frame = frame.merge(own, on=PARTICIPANT_KEY, how="left")

    partner = _swap_participant(own).rename(columns={c: f"other_{c}" for c in numeric})
    frame = frame.merge(partner, on=PARTICIPANT_KEY, how="left")
    frame = frame.fillna({c: 0.0 for c in [*numeric, *(f"other_{c}" for c in numeric)]})

    dialogs = messages.groupby("dialog_id", sort=False).agg(
        dialog_n_msgs=("message", "size"), last_speaker=("participant_index", "last")
    )
    frame = frame.merge(dialogs, left_on="dialog_id", right_index=True, how="left")
    frame["dialog_n_msgs"] = frame["dialog_n_msgs"].fillna(0)
    frame["msg_share"] = frame["n_msgs"] / frame["dialog_n_msgs"].clip(lower=1)
    frame["is_last_speaker"] = (frame["last_speaker"] == frame["participant_index"]).astype(float)
    frame = frame.drop(columns="last_speaker")
    for name in DIFF_FEATURES:
        frame[f"diff_{name}"] = frame[name] - frame[f"other_{name}"]

    texts = _participant_texts(messages, max_chars_per_message)
    frame = frame.merge(texts, on=PARTICIPANT_KEY, how="left")
    frame = frame.merge(
        _swap_participant(texts).rename(columns={"text_self": "text_other"}),
        on=PARTICIPANT_KEY,
        how="left",
    )
    for column in TEXT_COLUMNS:
        frame[column] = frame[column].fillna("").astype(object)
    return frame


def numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column not in NON_FEATURE_COLUMNS]
