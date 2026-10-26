"""HTTP-прототип сервиса: модель из MLflow Model Registry за FastAPI.

Запуск из папки `project/`:
    uv run --group serve uvicorn youarebot.api:app --port 8000
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_URI = os.environ.get("MODEL_URI", "models:/youarebot@champion")


class Message(BaseModel):
    message: int = Field(ge=0, description="Номер реплики в диалоге")
    participant_index: int = Field(ge=0, le=1)
    text: str = Field(max_length=10_000)


class DialogRequest(BaseModel):
    dialog_id: str
    messages: list[Message] = Field(min_length=1, max_length=500)


class ParticipantPrediction(BaseModel):
    participant_index: int
    is_bot: float = Field(ge=0, le=1, description="Вероятность того, что участник — бот")


class DialogResponse(BaseModel):
    dialog_id: str
    model_uri: str
    predictions: list[ParticipantPrediction]


def load_model(uri: str):
    import mlflow.pyfunc

    from youarebot.tracking import DEFAULT_TRACKING_URI

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    return mlflow.pyfunc.load_model(uri)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модель грузится один раз при старте, а не на каждый запрос.
    app.state.model = load_model(MODEL_URI)
    yield


app = FastAPI(title="YouAre.bot: бот или человек", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Обычный `def`: FastAPI выполнит инференс в пуле потоков и не заблокирует event loop.
@app.post("/predict", response_model=DialogResponse)
def predict(request: DialogRequest) -> DialogResponse:
    messages = pd.DataFrame([message.model_dump() for message in request.messages]).assign(
        dialog_id=request.dialog_id
    )
    predictions = app.state.model.predict(messages)
    return DialogResponse(
        dialog_id=request.dialog_id,
        model_uri=MODEL_URI,
        predictions=[
            ParticipantPrediction(participant_index=int(row.participant_index), is_bot=float(row.is_bot))
            for row in predictions.itertuples()
        ],
    )
