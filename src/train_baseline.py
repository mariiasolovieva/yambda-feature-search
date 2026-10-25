from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

RESULTS_DIR = Path("results")
TRAIN_PATH = RESULTS_DIR / "train.parquet"
VAL_PATH = RESULTS_DIR / "val.parquet"
TEST_PATH = RESULTS_DIR / "test.parquet"

MODEL_PATH = RESULTS_DIR / "catboost_baseline.cbm"
METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"
TEST_PREDS_PATH = RESULTS_DIR / "baseline_test_preds.parquet"
FEAT_IMP_PATH = RESULTS_DIR / "baseline_feature_importance.csv"

TARGET_COL = "label"

NON_FEATURE_COLS = {"uid", "item_id", "timestamp", "played_ratio_pct", "label"}

TOP_K = 10

CATBOOST_PARAMS = dict(
    iterations=2000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Logloss",
    random_seed=42,
    early_stopping_rounds=100,
    verbose=200,
    thread_count=-1,
)


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
    """LogLoss and ROC-AUC"""
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
    NDCG@K and Recall@K by user.

    For each user, we take his events in a split (these are the "candidates"),
    we sort by prediction and count NDCG/Recall in the top K.
    Users without positive examples are skipped in the split
    (NDCG and Recall are not defined for them)
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


def main() -> None:
    train = load_split(TRAIN_PATH)
    val = load_split(VAL_PATH)
    test = load_split(TEST_PATH)

    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    feature_cols = get_feature_columns(train)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")

    X_train = train[feature_cols]
    y_train = train[TARGET_COL].to_numpy()
    X_val = val[feature_cols]
    y_val = val[TARGET_COL].to_numpy()
    X_test = test[feature_cols]
    y_test = test[TARGET_COL].to_numpy()

    print(
        f"\nPositive rate: train={y_train.mean():.4f}, "
        f"val={y_val.mean():.4f}, test={y_test.mean():.4f}"
    )

    print("\n-->Training CatBoost")
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )

    print(f"\nBest iteration: {model.get_best_iteration()}")
    print(f"Best score: {model.get_best_score()}")

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    results = {}
    for name, y_true, pred, split_df in (
        ("val", y_val, val_pred, val),
        ("test", y_test, test_pred, test),
    ):
        print(f"\n=== {name} ===")
        clf = evaluate_classification(y_true, pred)
        rank = evaluate_ranking(split_df, pred, TOP_K)

        print(f"  logloss:  {clf['logloss']:.4f}")
        print(f"  roc_auc:  {clf['roc_auc']:.4f}")
        print(f"  ndcg@{TOP_K}:   {rank[f'ndcg@{TOP_K}']:.4f}")
        print(f"  recall@{TOP_K}: {rank[f'recall@{TOP_K}']:.4f}")
        print(f"  users with positives: {rank['n_users_with_positives']}")

        results[name] = {**clf, **rank}

    importance = model.get_feature_importance()
    feat_imp = (
        pd.DataFrame({"feature": feature_cols, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    print("\n=== Top-15 feature importance ===")
    print(feat_imp.head(15).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))
    print(f"\nSaved model: {MODEL_PATH}")

    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved metrics: {METRICS_PATH}")

    test_out = test[["uid", "item_id", "timestamp", "label"]].copy()
    test_out["pred_proba"] = test_pred
    test_out.to_parquet(TEST_PREDS_PATH, index=False)
    print(f"Saved test predictions: {TEST_PREDS_PATH}")

    feat_imp.to_csv(FEAT_IMP_PATH, index=False)
    print(f"Saved feature importance: {FEAT_IMP_PATH}")

    print("\n--> train_baseline done")


if __name__ == "__main__":
    main()

