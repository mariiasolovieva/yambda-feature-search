"""Ablation study для CatBoost baseline.

Обучаем одну и ту же CatBoost-модель с одинаковыми гиперпараметрами на разных наборах признаков, 
чтобы понять вклад каждой группы.

Конфигурации:
    full                      -- все 43 признака (совпадает с baseline)
    no_u_days_since_last_event -- без одного признака-сессионного индикатора
    no_recency                -- без всех *_days_since_* признаков
    no_user_features          -- без всех u_* (история пользователя)
    no_pair_features          -- без всех p_* (история пары user-item)
    only_ctx                  -- только 3 контекстных признака

Каждая конфигурация обучается на train, early stopping по val,
метрики считаются на val и test. Сравнительная таблица сохраняется
в results/ablation_comparison.csv, полные метрики — в
results/ablation_metrics.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

RESULTS_DIR = Path("results")
TRAIN_PATH = RESULTS_DIR / "train.parquet"
VAL_PATH = RESULTS_DIR / "val.parquet"
TEST_PATH = RESULTS_DIR / "test.parquet"

METRICS_PATH = RESULTS_DIR / "ablation_metrics.json"
COMPARISON_PATH = RESULTS_DIR / "ablation_comparison.csv"

TARGET_COL = "label"
TOP_K = 10

NON_FEATURE_COLS = {"uid", "item_id", "timestamp", "played_ratio_pct", "label"}

RECENCY_FEATURES = {
    "u_days_since_last_event",
    "u_days_since_first_event",
    "i_days_since_last_listen",
    "i_days_since_first_listen",
    "p_days_since_last_listen",
}

CATBOOST_PARAMS = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=42,
    early_stopping_rounds=100,
    verbose=0,
    thread_count=-1,
)


EXPERIMENTS = [
    ("full", lambda feats: feats),
    (
        "no_u_days_since_last_event",
        lambda feats: [f for f in feats if f != "u_days_since_last_event"],
    ),
    (
        "no_recency",
        lambda feats: [f for f in feats if f not in RECENCY_FEATURES],
    ),
    (
        "no_user_features",
        lambda feats: [f for f in feats if not f.startswith("u_")],
    ),
    (
        "no_pair_features",
        lambda feats: [f for f in feats if not f.startswith("p_")],
    ),
    (
        "only_ctx",
        lambda feats: [f for f in feats if f.startswith("ctx_")],
    ),
]


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Run src/split_data.py first."
        )
    return pd.read_parquet(path)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def evaluate_classification(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict:
    return {
        "logloss": float(log_loss(y_true, y_pred_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
    }


def evaluate_ranking(
    df: pd.DataFrame,
    y_pred_proba: np.ndarray,
    k: int,
) -> dict:
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


def run_experiment(
    name: str,
    feature_cols: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """Обучает CatBoost на заданном наборе фичей и возвращает метрики."""
    print(f"\n--- {name}: {len(feature_cols)} features ---")

    X_train = train[feature_cols]
    y_train = train[TARGET_COL].to_numpy()
    X_val = val[feature_cols]
    y_val = val[TARGET_COL].to_numpy()
    X_test = test[feature_cols]
    y_test = test[TARGET_COL].to_numpy()

    t0 = time.time()
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    elapsed = time.time() - t0

    best_iter = int(model.get_best_iteration())
    print(
        f"  best_iter={best_iter}, "
        f"val_logloss={model.get_best_score()['validation']['Logloss']:.4f}, "
        f"time={elapsed:.1f}s"
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    clf_val = evaluate_classification(y_val, val_pred)
    clf_test = evaluate_classification(y_test, test_pred)
    rank_val = evaluate_ranking(val, val_pred, TOP_K)
    rank_test = evaluate_ranking(test, test_pred, TOP_K)

    result = {
        "n_features": len(feature_cols),
        "best_iter": best_iter,
        "train_time_sec": round(elapsed, 2),
        "val": {**clf_val, **rank_val},
        "test": {**clf_test, **rank_test},
    }

    print(
        f"  val:  logloss={clf_val['logloss']:.4f}, "
        f"roc_auc={clf_val['roc_auc']:.4f}, "
        f"ndcg@{TOP_K}={rank_val[f'ndcg@{TOP_K}']:.4f}, "
        f"recall@{TOP_K}={rank_val[f'recall@{TOP_K}']:.4f}"
    )
    print(
        f"  test: logloss={clf_test['logloss']:.4f}, "
        f"roc_auc={clf_test['roc_auc']:.4f}, "
        f"ndcg@{TOP_K}={rank_test[f'ndcg@{TOP_K}']:.4f}, "
        f"recall@{TOP_K}={rank_test[f'recall@{TOP_K}']:.4f}"
    )

    return result



def build_comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rows.append({
            "experiment": name,
            "n_features": r["n_features"],
            "best_iter": r["best_iter"],
            "train_time_sec": r["train_time_sec"],
            "val_logloss": r["val"]["logloss"],
            "val_roc_auc": r["val"]["roc_auc"],
            "val_ndcg@10": r["val"][f"ndcg@{TOP_K}"],
            "test_logloss": r["test"]["logloss"],
            "test_roc_auc": r["test"]["roc_auc"],
            "test_ndcg@10": r["test"][f"ndcg@{TOP_K}"],
        })
    return pd.DataFrame(rows)


def add_deltas_vs_full(df: pd.DataFrame) -> pd.DataFrame:
    """add delta to 'full' by test_logloss & test_roc_auc."""
    full = df[df["experiment"] == "full"].iloc[0]
    df = df.copy()
    df["d_test_logloss"] = df["test_logloss"] - full["test_logloss"]
    df["d_test_roc_auc"] = df["test_roc_auc"] - full["test_roc_auc"]
    return df


def main() -> None:
    train = load_split(TRAIN_PATH)
    val = load_split(VAL_PATH)
    test = load_split(TEST_PATH)

    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    all_features = get_feature_columns(train)
    print(f"Full feature set: {len(all_features)} features")

    results: dict = {}

    for exp_name, transform in EXPERIMENTS:
        feats = transform(all_features)
        if not feats:
            print(f"\n[skip] {exp_name}: no features left")
            continue
        results[exp_name] = run_experiment(
            exp_name, feats, train, val, test
        )

    comparison = build_comparison_table(results)
    comparison = add_deltas_vs_full(comparison)

    print("\n=== Ablation comparison (test) ===")
    cols_to_show = [
        "experiment", "n_features",
        "test_logloss", "d_test_logloss",
        "test_roc_auc", "d_test_roc_auc",
        "test_ndcg@10",
    ]
    print(comparison[cols_to_show].to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {METRICS_PATH}")

    comparison.to_csv(COMPARISON_PATH, index=False)
    print(f"Saved comparison: {COMPARISON_PATH}")

    print("\n--> train_baseline_ablation done")


if __name__ == "__main__":
    main()
