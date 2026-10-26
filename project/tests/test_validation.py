import pandas as pd
import pandera.pandas as pa
import pytest

from tests.conftest import make_dialogs
from youarebot.data import Dataset, make_submission_id
from youarebot.validation import DataContractError, validate_dataset


def _dataset():
    train_messages, train_labels = make_dialogs(prefix="train")
    test_messages, test_labels = make_dialogs(n_dialogs=4, prefix="test")
    test_ids = test_labels.drop(columns="is_bot").assign(
        ID=lambda frame: make_submission_id(frame["dialog_id"], frame["participant_index"])
    )
    return Dataset(train_messages, train_labels, test_messages, test_ids)


def test_valid_dataset_passes():
    validate_dataset(_dataset())


def test_unknown_participant_index_fails():
    dataset = _dataset()
    dataset.train_labels.loc[0, "participant_index"] = 2
    with pytest.raises(pa.errors.SchemaError):
        validate_dataset(dataset)


def test_labels_for_missing_dialog_fail():
    dataset = _dataset()
    extra = dataset.train_labels.iloc[[0]].assign(dialog_id="no-such-dialog")
    broken = Dataset(
        dataset.train_messages,
        pd.concat([dataset.train_labels, extra], ignore_index=True),
        dataset.test_messages,
        dataset.test_ids,
    )
    with pytest.raises(DataContractError):
        validate_dataset(broken)
