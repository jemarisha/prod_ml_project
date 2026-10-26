"""Командная строка проекта: `uv run youarebot <команда>` из папки `project/`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from youarebot.config import load_config, load_project_settings


def _validate_data(args: argparse.Namespace) -> int:
    from youarebot.data import data_checksums, data_version, load_dataset, read_expected_checksums
    from youarebot.validation import data_quality_report, validate_dataset

    settings = load_project_settings()
    dataset = validate_dataset(load_dataset(settings.data))
    checksums = data_checksums(settings.data)
    print(f"Схемы и ссылочная целостность: OK. Версия данных: {data_version(checksums)}")
    for name, value in data_quality_report(dataset).items():
        print(f"  {name}: {value:g}")

    expected_path = settings.data.checksums_file
    if expected_path.exists():
        expected = read_expected_checksums(expected_path)
        mismatched = sorted(name for name, digest in checksums.items() if expected.get(name) != digest)
        if mismatched:
            print(f"ВНИМАНИЕ: файлы отличаются от {expected_path}: {mismatched}. "
                  "Результаты могут не совпасть с залогированными.")
        else:
            print(f"Контрольные суммы совпадают с {expected_path}")
    return 0


def _run(args: argparse.Namespace) -> int:
    from youarebot.train import run_experiment

    results = []
    for path in args.configs:
        config = load_config(path)
        print(f"==> {config.run_name}: {config.description}")
        result = run_experiment(config)
        print(f"    CV logloss = {result.cv_logloss:.4f}  (run_id {result.run_id})")
        results.append(result)

    print("\nИтог (чем меньше logloss, тем лучше):")
    for result in sorted(results, key=lambda item: item.cv_logloss):
        print(f"  {result.cv_logloss:.4f}  {result.run_name}")
    return 0


def _promote(args: argparse.Namespace) -> int:
    from youarebot.registry import promote_best_runs
    from youarebot.tracking import configure_mlflow

    settings = load_project_settings()
    configure_mlflow(settings.tracking.experiment_name)
    for item in promote_best_runs(
        settings.tracking.experiment_name, settings.tracking.registered_model_name, args.metric
    ):
        print(f"{settings.tracking.registered_model_name}@{item.alias}: версия {item.version}, "
              f"{item.run_name}, {args.metric} = {item.metric:.4f}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    from youarebot.predict import load_registered_model, predict_submission
    from youarebot.tracking import configure_mlflow

    settings = load_project_settings()
    configure_mlflow(settings.tracking.experiment_name)
    model = load_registered_model(settings.tracking.registered_model_name, args.alias)
    submission = predict_submission(model, settings.data, args.output)
    print(f"Сабмит {args.output}: {len(submission)} строк, средняя P(бот) = "
          f"{submission['is_bot'].mean():.3f}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    from youarebot.predict import load_registered_model, stream_dialog
    from youarebot.tracking import configure_mlflow

    settings = load_project_settings()
    configure_mlflow(settings.tracking.experiment_name)
    model = load_registered_model(settings.tracking.registered_model_name, args.alias)
    for participant, text, probability in stream_dialog(model, args.dialog):
        print(f"{participant}: {text}")
        print(f"is_bot_probability: {probability:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="youarebot", description="Классификатор «бот или человек»")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate-data", help="проверить контракт и версию данных")
    validate.set_defaults(handler=_validate_data)

    run = commands.add_parser("run", help="провести эксперименты и залогировать их в MLflow")
    run.add_argument("configs", nargs="+", type=Path, help="YAML-файлы из configs/experiments")
    run.set_defaults(handler=_run)

    promote = commands.add_parser("promote", help="назначить алиасы champion и challenger")
    promote.add_argument("--metric", default="cv_logloss")
    promote.set_defaults(handler=_promote)

    predict = commands.add_parser("predict", help="сабмит моделью из Model Registry")
    predict.add_argument("--alias", default="champion")
    predict.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    predict.set_defaults(handler=_predict)

    demo = commands.add_parser("demo", help="P(бот) после каждой реплики одного диалога")
    demo.add_argument("dialog", type=Path)
    demo.add_argument("--alias", default="champion")
    demo.set_defaults(handler=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    # Консоль Windows по умолчанию не в UTF-8, а в репликах есть кириллица и эмодзи.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
