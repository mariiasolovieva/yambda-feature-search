"""Автоматическая генерация и отбор признаков через grid search по DSL.

Идея:
    Определяем ограниченный язык описания агрегатов:
        (group_by, window_days, metric)
    где group_by in {(uid,), (item_id,), (uid, item_id)},
    window_days in {7, 30},
    metric in {listen_count, played_ratio_mean, like_count},
    плюс recency-спецификации (group_by times event_type).

    Материализуем все спецификации теми же примитивами, что и в
    build_features.py (защита от утечек гарантирована), обучаем CatBoost
    на всех авто-фичах, отбираем топ-K по важности, переобучаем и
    сравниваем с ручными baseline.

Артефакты:
    results/auto_features_metrics.json
    results/auto_features_comparison.csv
    results/auto_features_importance.csv
    results/auto_features_list.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import (  # noqa: E402
    load_events,
    prepare_events,
    _window_sum,
    _recency_days,
    _window_units,
)

RESULTS_DIR = Path("results")
EVENTS_PATH = Path("data/events_sample.parquet")
FEATURES_PATH = RESULTS_DIR / "features.parquet"

METRICS_PATH = RESULTS_DIR / "auto_features_metrics.json"
COMPARISON_PATH = RESULTS_DIR / "auto_features_comparison.csv"
IMPORTANCE_PATH = RESULTS_DIR / "auto_features_importance.csv"
SPECS_PATH = RESULTS_DIR / "auto_features_list.json"

TARGET_COL = "label"
TOP_K_RANKING = 10
TOP_K_SELECT = 15

# grid search parameters
WINDOWS_DAYS = (1, 3, 7, 14, 30)
GROUP_BYS = (("uid",), ("item_id",), ("uid", "item_id"))
SOURCE_METRICS = (
    ("listen_count", "is_listen", "sum"),
    ("played_ratio_mean", "played_ratio_x_listen", "mean"),
    ("like_count", "is_like", "sum"),
)

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

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


# DSL

def enumerate_specs() -> list[dict]:
    specs: list[dict] = []

    for group_by in GROUP_BYS:
        group_tag = "_".join(group_by)
        for window_days in WINDOWS_DAYS:
            for metric_name, source_col, agg in SOURCE_METRICS:
                specs.append({
                    "name": f"auto_{group_tag}_{metric_name}_{window_days}d",
                    "group_by": list(group_by),
                    "window_days": window_days,
                    "metric_name": metric_name,
                    "source_col": source_col,
                    "agg": agg,
                })

    for group_by in GROUP_BYS:
        group_tag = "_".join(group_by)
        for evt in ("all", "listen", "like"):
            specs.append({
                "name": f"auto_{group_tag}_days_since_{evt}",
                "group_by": list(group_by),
                "recency": True,
                "evt_type": evt,
            })

    return specs


def materialize_spec(
    spec: dict,
    listens: pd.DataFrame,
    events: pd.DataFrame,
) -> np.ndarray:
    """Counts one column according to the specification for all listens"""
    if spec.get("recency"):
        source = events
        if spec["evt_type"] == "listen":
            source = events[events["event_type"] == "listen"]
        elif spec["evt_type"] == "like":
            source = events[events["event_type"] == "like"]
        return _recency_days(listens, source, spec["group_by"])

    wu = _window_units(spec["window_days"])

    if spec["agg"] == "sum":
        return _window_sum(
            listens, events, spec["group_by"], wu, spec["source_col"]
        )

    # mean = sum(played_ratio) / count(listens)
    num = _window_sum(
        listens, events, spec["group_by"], wu, spec["source_col"]
    )
    den = _window_sum(
        listens, events, spec["group_by"], wu, "is_listen"
    )
    return np.where(den > 0, num / den, 0.0)

def analyze_window_sensitivity(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    auto_cols: list[str],
) -> dict:
    """
    Для каждой пары (group_by, metric) обучает однофичевые модели
    по каждому окну из WINDOWS_DAYS. Показывает, какое окно даёт
    минимальный LogLoss на validation — это и есть автоматический
    выбор окна по данным.
    """
    results: dict = {}

    for group_by in GROUP_BYS:
        group_tag = "_".join(group_by)
        for metric_name, _, _ in SOURCE_METRICS:
            per_window: dict = {}
            for window_days in WINDOWS_DAYS:
                col = f"auto_{group_tag}_{metric_name}_{window_days}d"
                if col not in auto_cols:
                    continue

                X_tr = train_df[[col]].fillna(0.0)
                X_va = val_df[[col]].fillna(0.0)
                X_te = test_df[[col]].fillna(0.0)
                y_tr = train_df[TARGET_COL].to_numpy()
                y_va = val_df[TARGET_COL].to_numpy()
                y_te = test_df[TARGET_COL].to_numpy()

                _, res = train_and_eval(
                    X_tr, y_tr,
                    X_va, y_va, val_df["uid"].to_numpy(),
                    X_te, y_te, test_df["uid"].to_numpy(),
                )
                per_window[window_days] = {
                    "val_logloss": res["val"]["logloss"],
                    "test_logloss": res["test"]["logloss"],
                    "test_roc_auc": res["test"]["roc_auc"],
                }

            if not per_window:
                continue

            best_w = min(per_window, key=lambda w: per_window[w]["val_logloss"])
            results[f"{group_tag}_{metric_name}"] = {
                "per_window": per_window,
                "best_window_days": best_w,
                "best_val_logloss": per_window[best_w]["val_logloss"],
            }

    return results


def build_auto_features(
    listens: pd.DataFrame,
    events: pd.DataFrame,
    specs: list[dict],
) -> pd.DataFrame:
    print(f"Materializing {len(specs)} auto-features...")
    result = pd.DataFrame(index=listens.index)
    for i, spec in enumerate(specs, 1):
        result[spec["name"]] = materialize_spec(spec, listens, events)
        if i % 10 == 0:
            print(f"  {i}/{len(specs)} done")
    return result


# evaluation

def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uids: np.ndarray,
    k: int,
) -> dict:
    ll = float(log_loss(y_true, y_pred))
    auc = float(roc_auc_score(y_true, y_pred))

    df = pd.DataFrame({"uid": uids, "y": y_true, "p": y_pred})
    ndcgs, recalls = [], []

    for _, group in df.groupby("uid"):
        y = group["y"].to_numpy()
        n_pos = int(y.sum())
        if n_pos == 0:
            continue
        order = np.argsort(-group["p"].to_numpy())
        y_sorted = y[order][:k]
        discounts = 1.0 / np.log2(np.arange(2, len(y_sorted) + 2))
        dcg = float((y_sorted * discounts).sum())
        idcg = float(discounts[: min(n_pos, k)].sum())
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        recalls.append(int(y_sorted.sum()) / n_pos)

    return {
        "logloss": ll,
        "roc_auc": auc,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
    }


def train_and_eval(
    X_train, y_train,
    X_val, y_val, val_uids,
    X_test, y_test, test_uids,
) -> tuple[CatBoostClassifier, dict]:
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True,
    )
    elapsed = time.time() - t0

    val_pred = model.predict_proba(X_val)[:, 1]
    test_pred = model.predict_proba(X_test)[:, 1]

    return model, {
        "best_iter": int(model.get_best_iteration()),
        "time_sec": round(elapsed, 2),
        "val": evaluate(y_val, val_pred, val_uids, TOP_K_RANKING),
        "test": evaluate(y_test, test_pred, test_uids, TOP_K_RANKING),
    }



def main() -> None:
    events = load_events(EVENTS_PATH)
    events = prepare_events(events)

    features = pd.read_parquet(FEATURES_PATH)
    print(f"Features (listens): {features.shape}")

    listens = features[
        ["uid", "item_id", "timestamp", TARGET_COL]
    ].reset_index(drop=True)

    specs = enumerate_specs()
    print(f"Enumerated {len(specs)} DSL specs")

    auto_feats = build_auto_features(listens, events, specs)
    assert len(auto_feats) == len(features), "row count mismatch"

    features_aug = pd.concat(
        [features.reset_index(drop=True), auto_feats.reset_index(drop=True)],
        axis=1,
    )

    # like in split_data.py
    train_thr = int(features_aug["timestamp"].quantile(TRAIN_FRAC))
    val_thr = int(
        features_aug["timestamp"].quantile(TRAIN_FRAC + VAL_FRAC)
    )
    train_df = features_aug[features_aug["timestamp"] < train_thr].copy()
    val_df = features_aug[
        (features_aug["timestamp"] >= train_thr)
        & (features_aug["timestamp"] < val_thr)
    ].copy()
    test_df = features_aug[features_aug["timestamp"] >= val_thr].copy()

    print(
        f"Split sizes: train={len(train_df):,}, "
        f"val={len(val_df):,}, test={len(test_df):,}"
    )

    auto_cols = [s["name"] for s in specs]

    X_train = train_df[auto_cols].fillna(0.0)
    X_val = val_df[auto_cols].fillna(0.0)
    X_test = test_df[auto_cols].fillna(0.0)
    y_train = train_df[TARGET_COL].to_numpy()
    y_val = val_df[TARGET_COL].to_numpy()
    y_test = test_df[TARGET_COL].to_numpy()

    print(f"\n=== Stage 1: all {len(auto_cols)} auto-features ===")
    model_all, res_all = train_and_eval(
        X_train, y_train,
        X_val, y_val, val_df["uid"].to_numpy(),
        X_test, y_test, test_df["uid"].to_numpy(),
    )
    print(
        f"  best_iter={res_all['best_iter']} "
        f"time={res_all['time_sec']}s"
    )
    print(
        f"  val:  logloss={res_all['val']['logloss']:.4f}, "
        f"roc_auc={res_all['val']['roc_auc']:.4f}"
    )
    print(
        f"  test: logloss={res_all['test']['logloss']:.4f}, "
        f"roc_auc={res_all['test']['roc_auc']:.4f}"
    )

    importance = (
        pd.DataFrame({
            "feature": auto_cols,
            "importance": model_all.get_feature_importance(),
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(IMPORTANCE_PATH, index=False)

    print("\nTop-10 auto-features:")
    print(importance.head(10).to_string(index=False))

    selected = importance.head(TOP_K_SELECT)["feature"].tolist()

    print(f"\n=== Stage 2: selected top-{TOP_K_SELECT} ===")
    model_sel, res_sel = train_and_eval(
        X_train[selected], y_train,
        X_val[selected], y_val, val_df["uid"].to_numpy(),
        X_test[selected], y_test, test_df["uid"].to_numpy(),
    )
    print(
        f"  best_iter={res_sel['best_iter']} "
        f"time={res_sel['time_sec']}s"
    )
    print(
        f"  val:  logloss={res_sel['val']['logloss']:.4f}, "
        f"roc_auc={res_sel['val']['roc_auc']:.4f}"
    )
    print(
        f"  test: logloss={res_sel['test']['logloss']:.4f}, "
        f"roc_auc={res_sel['test']['roc_auc']:.4f}"
    )

    print("\n=== Window sensitivity (single-feature models) ===")
    window_analysis = analyze_window_sensitivity(
        train_df, val_df, test_df, auto_cols
    )
    for key, info in window_analysis.items():
        per_w = info["per_window"]
        parts = "  ".join(
            f"{w}d={per_w[w]['val_logloss']:.4f}"
            for w in WINDOWS_DAYS
            if w in per_w
        )
        print(
            f"  {key:>35}  best={info['best_window_days']:>2}d  "
            f"[{parts}]"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "n_specs": len(specs),
        "n_selected": TOP_K_SELECT,
        "stage1_all_features": res_all,
        "stage2_selected": res_sel,
        "selected_features": selected,
        "window_sensitivity": window_analysis,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    with open(SPECS_PATH, "w") as f:
        json.dump({"specs": specs, "selected": selected}, f, indent=2)

    comparison = pd.DataFrame([
        {
            "approach": "baseline_full (manual + session)",
            "test_logloss": 0.2495, "test_roc_auc": 0.9578,
            "test_ndcg@10": 0.9779,
        },
        {
            "approach": "baseline_honest (manual, no session)",
            "test_logloss": 0.4256, "test_roc_auc": 0.8446,
            "test_ndcg@10": 0.8477,
        },
        {
            "approach": f"auto_all ({len(auto_cols)} specs)",
            "test_logloss": res_all["test"]["logloss"],
            "test_roc_auc": res_all["test"]["roc_auc"],
            "test_ndcg@10": res_all["test"][f"ndcg@{TOP_K_RANKING}"],
        },
        {
            "approach": f"auto_selected (top-{TOP_K_SELECT})",
            "test_logloss": res_sel["test"]["logloss"],
            "test_roc_auc": res_sel["test"]["roc_auc"],
            "test_ndcg@10": res_sel["test"][f"ndcg@{TOP_K_RANKING}"],
        },
    ])
    comparison.to_csv(COMPARISON_PATH, index=False)

    print("\n Comparison (test)")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {METRICS_PATH}, {COMPARISON_PATH}, "
          f"{IMPORTANCE_PATH}, {SPECS_PATH}")
    print("--> auto_features done")

if __name__ == "__main__":
    main()
