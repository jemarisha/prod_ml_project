# Данные

Данные соревнования [You Are Bot](https://stackmorelayers.ai/competitions/you-are-bot)
доступны только зарегистрированным участникам, поэтому в репозиторий не коммитятся.

1. Зарегистрируйтесь в соревновании и скачайте файлы со вкладки «Данные».
2. Положите их в `data/raw/`:

```
data/raw/
├── train.json              # {dialog_id: [{text, message, participant_index}]}
├── ytrain.csv              # dialog_id, participant_index, is_bot
├── test.json
├── ytest.csv               # dialog_id, participant_index, ID
└── sample_submission.csv   # ID, is_bot
```

3. Проверьте, что версия данных та же, на которой получены результаты:

```bash
uv run youarebot validate-data
```

Ожидаемые SHA-256 лежат в `checksums.sha256`. Они же вместе с коротким
отпечатком `data_version` логируются в MLflow в каждом запуске.
