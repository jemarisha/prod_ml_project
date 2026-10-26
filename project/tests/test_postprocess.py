import numpy as np
import pandas as pd
import pytest

from youarebot.postprocess import exclusive_bot_normalize

PAIR = pd.DataFrame({"dialog_id": ["a", "a"], "participant_index": [0, 1]})


def test_two_confident_bots_are_not_allowed():
    result = exclusive_bot_normalize(PAIR, np.array([0.9, 0.9]))
    assert result.sum() < 1.0
    assert result[0] == pytest.approx(result[1])


def test_human_partner_keeps_bot_probability_high():
    result = exclusive_bot_normalize(PAIR, np.array([0.8, 0.01]))
    assert result[0] == pytest.approx(0.8 * 0.99 / (1 - 0.8 * 0.01))
    assert result[1] < 0.01


def test_participant_without_partner_is_unchanged():
    single = pd.DataFrame({"dialog_id": ["a"], "participant_index": [0]})
    assert exclusive_bot_normalize(single, np.array([0.3]))[0] == pytest.approx(0.3)


def test_output_is_a_valid_probability():
    rng = np.random.default_rng(0)
    keys = pd.DataFrame({"dialog_id": np.repeat(np.arange(50), 2), "participant_index": [0, 1] * 50})
    result = exclusive_bot_normalize(keys, rng.uniform(0, 1, size=100))
    assert ((result > 0) & (result < 1)).all()
