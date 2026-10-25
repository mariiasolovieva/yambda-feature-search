"""End-to-end pipeline for the yambda-feature-search project.

Запустите этот файл, чтобы прогнать весь собранный пайплайн целиком.

Шаги:
    1. download_data         -> data/events_sample.parquet
                                (пропускается, если файл уже есть)
    2. analyze_played_ratio  -> анализ target-переменной
    3. build_features        -> results/features.parquet
    4. split_data            -> results/{train,val,test}.parquet
    5. train_baseline_dummy  -> результаты dummy и oracle
    6. train_baseline        -> CatBoost на ручных признаках
    7. train_baseline_ablation -> ablation по группам признаков
    8. auto_features         -> автоматическая генерация признаков
                                и автоподбор временного окна
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


EVENTS_PATH = Path("data/events_sample.parquet")


STEPS = [
    {
        "name": "1/8. Download events sample",
        "script": "src/download_data.py",
        "skip_if": EVENTS_PATH,
    },
    {
        "name": "2/8. Analyze played_ratio_pct (target threshold)",
        "script": "src/analyze_played_ratio.py",
        "skip_if": None,
    },
    {
        "name": "3/8. Build features",
        "script": "src/build_features.py",
        "skip_if": None,
    },
    {
        "name": "4/8. Split by time",
        "script": "src/split_data.py",
        "skip_if": None,
    },
    {
        "name": "5/8. Naive baselines (dummy + oracle)",
        "script": "src/train_baseline_dummy.py",
        "skip_if": None,
    },
    {
        "name": "6/8. CatBoost baseline",
        "script": "src/train_baseline.py",
        "skip_if": None,
    },
    {
        "name": "7/8. Ablation over feature groups",
        "script": "src/train_baseline_ablation.py",
        "skip_if": None,
    },
    {
        "name": "8/8. Auto feature generation + window selection",
        "script": "src/auto_features.py",
        "skip_if": None,
    },
]


def run_script(script_path: str) -> None:
    """Запускает шаг пайплайна в отдельном процессе."""
    banner = "=" * 70
    print(f"\n{banner}\n  Running: {script_path}\n{banner}")
    subprocess.run([sys.executable, script_path], check=True)


def main() -> None:
    for step in STEPS:
        skip_if = step["skip_if"]
        if skip_if is not None and skip_if.exists():
            print(f"\n[skip] {step['name']}: {skip_if} already exists")
            continue

        print(f"\n{step['name']}")
        run_script(step["script"])

    banner = "=" * 70
    print(f"\n{banner}\n  --> run_exp done\n{banner}")


if __name__ == "__main__":
    main()
