"""Naive baselines: DummyClassifier(prior) и Oracle (верхняя граница).

Запуск:
    python src/baseline_dummy.py

Что считаем:
    - DummyClassifier(strategy="prior") — всегда предсказывает тренировочное
      распределение классов. Это «пол» качества: модель без обучения.
    - Oracle — идеальное ранжирование по истинным label. Это «потолок»
      качества: верхняя граница для ranking-метрик.
    - Теоретический random NDCG@K = pos_rate — для калибровки ожиданий.

Результаты сохраняются в results/dummy_metrics.json, dummy_test_preds.parquet.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import log_loss, roc_auc_score

RESULTS_DIR = Path("results")
TRAIN_PATH = RESULTS_DIR / "train.parquet"
VAL_PATH = RESULTS_DIR / "val.parquet"
TEST_PATH = RESULTS_DIR / "test.parquet"

METRICS_PATH = RESULTS_DIR / "dummy_metrics.json"
TEST_PREDS_PATH = RESULTS_DIR / "dummy_test_preds.parquet"

TARGET_COL = "label"
TOP_K = 10
RANDOM_SEED = 42


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Run src/split_data.py first."
        )
    return pd.read_parquet(path)


def evaluate_classification(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    """LogLoss и ROC-AUC. Для константного предиктора ROC-AUC = 0.5."""
    return {
        "logloss": float(log_loss(y_true, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
    }


def evaluate_ranking(
    df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    k: int,
) -> dict:
    """
    NDCG@K и Recall@K по пользователям.

    Для каждого пользователя берём его события в сплите (кандидаты),
    сортируем по предсказанию и считаем NDCG/Recall в топ-K.
    Пользователи без положительных примеров пропускаются.
    """
    df = df[["uid", "label"]].copy()
    df["_pred"] = y_pred_proba

    ndcgs = []
    recalls = []

    for _, group in df.groupby("uid"):
        y = group["label"].to_numpy()
        n_pos = int(y.sum())
        if n_pos == 0:
            continue

        p = group["_pred"].to_numpy()
        order = np.argsort(-p)
        y_sorted = y[order][:k]

        discounts = 1.0 / np.log2(np.arange(2, len(y_sorted) + 2))
        dcg = float((y_sorted * discounts).sum())

        n_pos_in_top = min(n_pos, k)
        idcg = float(discounts[:n_pos_in_top].sum())
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        hits = int(y_sorted.sum())
        recalls.append(hits / n_pos)

    return {
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        "n_users_with_positives": len(ndcgs),
    }


def oracle_predictions(y_true: np.ndarray) -> np.ndarray:
    """
    Идеальное ранжирование: все позитивы выше всех негативов.

    Добавляем случайный шум, чтобы разорвать связи внутри позитивов
    и внутри негативов (иначе argsort вернёт произвольный порядок).
    """
    rng = np.random.default_rng(RANDOM_SEED)
    noise = rng.uniform(0.0, 0.1, size=len(y_true))
    return y_true.astype("float64") + noise


def main() -> None:
    train = load_split(TRAIN_PATH)
    val = load_split(VAL_PATH)
    test = load_split(TEST_PATH)

    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL].to_numpy()

    dummy = DummyClassifier(strategy="prior", random_state=RANDOM_SEED)
    dummy.fit(X_train, y_train)

    class_prior = dummy.class_prior_
    print(
        f"\nDummy train prior: neg={class_prior[0]:.4f}, "
        f"pos={class_prior[1]:.4f}"
    )

    results: dict = {
        "train_prior_pos": float(y_train.mean()),
    }

    for name, split_df in (("val", val), ("test", test)):
        print(f"\n=== {name} ===")
        X = split_df.drop(columns=[TARGET_COL])
        y = split_df[TARGET_COL].to_numpy()

        # --- Dummy ---
        pred_dummy = dummy.predict_proba(X)[:, 1]
        clf_dummy = evaluate_classification(y, pred_dummy)
        rank_dummy = evaluate_ranking(split_df, pred_dummy, TOP_K)

        print(
            f"  dummy:  logloss={clf_dummy['logloss']:.4f}, "
            f"roc_auc={clf_dummy['roc_auc']:.4f}, "
            f"ndcg@{TOP_K}={rank_dummy[f'ndcg@{TOP_K}']:.4f}, "
            f"recall@{TOP_K}={rank_dummy[f'recall@{TOP_K}']:.4f}"
        )

        # --- Oracle ---
        pred_oracle = oracle_predictions(y)
        rank_oracle = evaluate_ranking(split_df, pred_oracle, TOP_K)

        print(
            f"  oracle: ndcg@{TOP_K}={rank_oracle[f'ndcg@{TOP_K}']:.4f}, "
            f"recall@{TOP_K}={rank_oracle[f'recall@{TOP_K}']:.4f} "
            f"(users with positives: {rank_oracle['n_users_with_positives']})"
        )

        pos_rate = float(y.mean())
        print(
            f"  reference: random ndcg@{TOP_K} ~= pos_rate = {pos_rate:.4f}"
        )

        results[name] = {
            "dummy": {**clf_dummy, **rank_dummy},
            "oracle": rank_oracle,
            "pos_rate": pos_rate,
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {METRICS_PATH}")

    test_out = test[["uid", "item_id", "timestamp", "label"]].copy()
    test_out["pred_proba"] = dummy.predict_proba(
        test.drop(columns=[TARGET_COL])
    )[:, 1]
    test_out.to_parquet(TEST_PREDS_PATH, index=False)
    print(f"Saved test predictions: {TEST_PREDS_PATH}")

    print("\n--> baseline_dummy done")


if __name__ == "__main__":
    main()
    