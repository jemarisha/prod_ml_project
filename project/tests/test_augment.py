import numpy as np

from youarebot.augment import make_prefixes, source_dialog
from youarebot.config import ModelConfig
from youarebot.features import build_participant_features, numeric_feature_columns
from youarebot.models import build_model
from youarebot.train import Augmentation, cross_validate_oof


def test_prefixes_inherit_labels_of_their_dialog(synthetic):
    messages, labels = synthetic
    prefixes = make_prefixes(messages, labels, per_dialog=2, seed=0)

    lengths = prefixes.messages.groupby("dialog_id").size()
    assert lengths.between(2, 5).all()  # в синтетических диалогах по 6 реплик
    merged = prefixes.labels.assign(source=source_dialog(prefixes.labels["dialog_id"])).merge(
        labels.rename(columns={"dialog_id": "source"}), on=["source", "participant_index"]
    )
    assert (merged["is_bot_x"] == merged["is_bot_y"]).all()


def test_cv_uses_prefixes_only_from_training_dialogs(synthetic, monkeypatch):
    messages, labels = synthetic
    features = build_participant_features(messages, participants=labels)
    target = labels["is_bot"].to_numpy()
    prefixes = make_prefixes(messages, labels, per_dialog=2, seed=0)
    prefix_features = build_participant_features(prefixes.messages, participants=prefixes.labels)
    augmentation = Augmentation(
        features=prefix_features,
        target=prefixes.labels["is_bot"].to_numpy(),
        weight=np.full(len(prefixes.labels), 0.5),
        source=source_dialog(prefix_features["dialog_id"]).to_numpy(),
    )
    seen_sources: list[set[str]] = []
    original_fit = type(build_model(ModelConfig(type="constant"), [], 0)).fit

    def recording_fit(self, X, y, sample_weight=None):
        seen_sources.append(set(source_dialog(X["dialog_id"])))
        return original_fit(self, X, y, sample_weight=sample_weight)

    model = build_model(ModelConfig(type="constant"), numeric_feature_columns(features), 0)
    monkeypatch.setattr(type(model), "fit", recording_fit)
    _, folds = cross_validate_oof(model, features, target, 3, 0, augmentation)

    for fold, sources in enumerate(seen_sources):
        valid_dialogs = set(features.loc[folds == fold, "dialog_id"])
        assert not sources & valid_dialogs
