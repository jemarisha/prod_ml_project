import pytest

from tests.conftest import PROJECT_ROOT
from youarebot.config import ModelConfig, TfidfConfig, load_config
from youarebot.features import build_participant_features, numeric_feature_columns
from youarebot.inference import positive_proba
from youarebot.models import build_model
from youarebot.pyfunc_model import DialogBotDetector

FAST_CATBOOST = {"iterations": 20, "thread_count": 1}
TFIDF = [TfidfConfig(column="text_self", analyzer="char_wb", ngram_range=(2, 3), min_df=1)]

MODEL_CONFIGS = {
    "constant": ModelConfig(type="constant"),
    "logreg": ModelConfig(type="logreg", tfidf=TFIDF),
    "catboost": ModelConfig(type="catboost", params=FAST_CATBOOST),
    "catboost_text": ModelConfig(
        type="catboost", text_columns=["text_self", "text_other"], params=FAST_CATBOOST
    ),
    "blend": ModelConfig(
        type="blend",
        components=[
            ModelConfig(type="logreg", tfidf=TFIDF),
            ModelConfig(type="catboost", params=FAST_CATBOOST),
        ],
    ),
}


def _fit(config: ModelConfig, messages, labels):
    features = build_participant_features(messages, participants=labels)
    model = build_model(config, numeric_feature_columns(features), seed=0)
    return model.fit(features, labels["is_bot"].to_numpy()), features


@pytest.mark.parametrize("name", MODEL_CONFIGS)
def test_model_predicts_probabilities(name, synthetic):
    messages, labels = synthetic
    model, features = _fit(MODEL_CONFIGS[name], messages, labels)
    proba = positive_proba(model, features)
    assert proba.shape == (len(labels),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_mlflow_wrapper_scores_both_participants_of_each_dialog(synthetic):
    messages, labels = synthetic
    model, _ = _fit(MODEL_CONFIGS["logreg"], messages, labels)
    result = DialogBotDetector(model, max_chars_per_message=1000, exclusive_bot=True).predict(
        None, messages.sample(frac=1, random_state=0)
    )
    assert list(result.columns) == ["dialog_id", "participant_index", "is_bot"]
    assert len(result) == 2 * messages["dialog_id"].nunique()
    per_dialog = result.groupby("dialog_id")["is_bot"].sum()
    assert (per_dialog < 1).all()


def test_override_changes_param_and_run_name():
    config = load_config(
        PROJECT_ROOT / "configs" / "experiments" / "03_logreg_tfidf_features.yaml",
        base_path=PROJECT_ROOT / "configs" / "base.yaml",
        overrides=["model.params.C=0.25", "cv.n_splits=3"],
    )
    assert config.model.params["C"] == 0.25
    assert config.cv.n_splits == 3
    assert config.run_name == "03_logreg_tfidf_features[C=0.25,n_splits=3]"


@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml")))
def test_experiment_configs_are_valid(path):
    config = load_config(path, base_path=PROJECT_ROOT / "configs" / "base.yaml")
    assert config.run_name
    build_model(config.model, numeric_columns=["n_msgs"], seed=config.seed)
