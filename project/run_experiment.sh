#!/usr/bin/env bash
# Сквозной воспроизводимый запуск одной командой:
#   окружение из uv.lock → тесты → проверка данных → эксперименты в MLflow
#   → champion/challenger в Model Registry → сабмит → демо на одном диалоге.
#
# Использование (из любой папки):
#   ./run_experiment.sh                                    # все конфиги из configs/experiments
#   ./run_experiment.sh configs/experiments/05_catboost_text.yaml   # выбранные конфиги
set -euo pipefail
cd "$(dirname "$0")"

export PYTHONUTF8=1               # консоль Windows не в UTF-8 по умолчанию
export PYTHONHASHSEED=0           # детерминированный порядок в set/dict строк
export MLFLOW_DISABLE_AGENT_HINT=1
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"

if [ "$#" -gt 0 ]; then
    configs=("$@")
else
    configs=(configs/experiments/*.yaml)
fi

echo "==> Окружение: uv sync --locked"
uv sync --locked

echo "==> Тесты"
uv run pytest -q

echo "==> Контракт и версия данных"
uv run youarebot validate-data

echo "==> Эксперименты (MLflow: ${MLFLOW_TRACKING_URI})"
uv run youarebot run "${configs[@]}"

if [ "$#" -eq 0 ]; then
    echo "==> Подбор гиперпараметров: каждый вариант — отдельный запуск в MLflow"
    for c in 0.25 0.5 1 4; do
        uv run youarebot run configs/experiments/03_logreg_tfidf_features.yaml --set "model.params.C=${c}"
    done
    for depth in 4 8; do
        uv run youarebot run configs/experiments/07_catboost_text_light.yaml --set "model.params.depth=${depth}"
    done
fi

echo "==> Model Registry: champion и challenger"
uv run youarebot promote

echo "==> Сабмит моделью champion"
uv run youarebot predict --alias champion --output submissions/submission.csv

echo "==> Демо: P(бот) после каждой реплики"
uv run youarebot demo tests/data/sample_dialog.json

echo "Готово. Интерфейс MLflow: uv run mlflow ui --backend-store-uri ${MLFLOW_TRACKING_URI}"
