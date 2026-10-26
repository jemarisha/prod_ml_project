import pytest

from youarebot.data import dialogs_to_frame
from youarebot.features import build_participant_features, is_doubled, numeric_feature_columns


def _dialog(*texts_by_participant: tuple[int, str]):
    return dialogs_to_frame(
        {
            "d": [
                {"text": text, "message": i, "participant_index": p}
                for i, (p, text) in enumerate(texts_by_participant)
            ]
        }
    )


def test_echo_of_previous_message_is_detected():
    features = build_participant_features(_dialog((0, "Привет"), (1, "привет "), (0, "как дела")))
    by_participant = features.set_index("participant_index")
    assert by_participant.loc[1, "is_echo_share"] == 1.0
    assert by_participant.loc[0, "is_echo_share"] == 0.0
    assert by_participant.loc[0, "other_is_echo_share"] == 1.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("норм\n\nнорм", True),
        ("saman saman \n\nsaman saman", True),
        ("норм\n\nне норм", False),
        ("просто текст", False),
    ],
)
def test_doubled_message(text, expected):
    assert is_doubled(text) is expected


def test_participant_without_messages_gets_zero_features():
    features = build_participant_features(_dialog((0, "эй")))
    silent = features[features["participant_index"] == 1].iloc[0]
    assert silent["n_msgs"] == 0
    assert silent["text_self"] == ""
    assert silent["text_other"] == "эй"


def test_cyrillic_letters_are_counted():
    # Регрессия: у строк на pyarrow `\w` — только ASCII, кириллица выпадала из подсчёта.
    features = build_participant_features(_dialog((0, "ПРИВЕТ"), (1, "привет")))
    by_participant = features.set_index("participant_index")
    assert by_participant.loc[0, "upper_ratio_mean"] == 1.0
    assert by_participant.loc[1, "upper_ratio_mean"] == 0.0


def test_turn_taking_features():
    features = build_participant_features(
        _dialog((0, "привет"), (1, "привет, как дела?"), (0, "ок"), (0, "ты тут?"))
    )
    by_participant = features.set_index("participant_index")
    # Участник 0 получил ответ на 1 реплику из 3; участник 1 — на свою единственную.
    assert by_participant.loc[0, "gets_reply_share"] == pytest.approx(1 / 3)
    assert by_participant.loc[1, "gets_reply_share"] == 1.0
    assert by_participant.loc[0, "max_run"] == 2
    assert by_participant.loc[1, "reply_len_ratio_mean"] > 0  # ответил длиннее, чем ему написали
    assert by_participant.loc[0, "first_is_greeting"] == 1.0


def test_rows_follow_given_participants_order(synthetic):
    messages, labels = synthetic
    shuffled = labels.sample(frac=1, random_state=0)
    features = build_participant_features(messages, participants=shuffled)
    assert list(zip(features["dialog_id"], features["participant_index"], strict=True)) == list(
        zip(shuffled["dialog_id"], shuffled["participant_index"], strict=True)
    )


def test_numeric_features_have_no_missing_values(synthetic):
    messages, _ = synthetic
    features = build_participant_features(messages)
    assert not features[numeric_feature_columns(features)].isna().any().any()
