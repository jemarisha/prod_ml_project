import pandas as pd
import pytest
from fastapi.testclient import TestClient

from youarebot import api


class ConstantModel:
    """Подмена MLflow-модели: тест проверяет HTTP-контракт, а не качество."""

    def predict(self, messages: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "dialog_id": messages["dialog_id"].iloc[0],
                "participant_index": [0, 1],
                "is_bot": [0.2, 0.7],
            }
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda uri: ConstantModel())
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_probability_per_participant(client):
    response = client.post(
        "/predict",
        json={
            "dialog_id": "d1",
            "messages": [
                {"message": 0, "participant_index": 0, "text": "привет"},
                {"message": 1, "participant_index": 1, "text": "привет"},
            ],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dialog_id"] == "d1"
    assert [p["participant_index"] for p in body["predictions"]] == [0, 1]


@pytest.mark.parametrize(
    "payload",
    [
        {"dialog_id": "d1", "messages": []},
        {"dialog_id": "d1", "messages": [{"message": 0, "participant_index": 2, "text": "эй"}]},
    ],
)
def test_invalid_request_is_rejected_with_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422
