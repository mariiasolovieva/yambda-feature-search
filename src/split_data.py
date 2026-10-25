from pathlib import Path

import pandas as pd

FEATURES_PATH = Path("results/features.parquet")
OUTPUT_DIR = Path("results")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# Ниже этого числа уникальных пользователей в test ranking-метрики
# (NDCG@K, Recall@K) будут шумными — печатаем предупреждение.
MIN_TEST_USERS_WARN = 30

# Если pos_rate в test отклоняется от train больше чем на это значение —
# предупреждаем о временном дрейфе.
POS_RATE_DRIFT_WARN = 0.05


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}. Run src/build_features.py first."
        )

    features = pd.read_parquet(path)

    required = {"uid", "item_id", "timestamp", "label"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return features


def split_by_time(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Глобальный временной сплит по квантилям timestamp.

    Границы выбираются по квантилям, а не по фиксированным датам,
    чтобы доли train / val / test были стабильны независимо от
    распределения событий по времени.
    """
    train_thr = int(df["timestamp"].quantile(train_frac))
    val_thr = int(df["timestamp"].quantile(train_frac + val_frac))

    train = df[df["timestamp"] < train_thr].copy()
    val = df[
        (df["timestamp"] >= train_thr) & (df["timestamp"] < val_thr)
    ].copy()
    test = df[df["timestamp"] >= val_thr].copy()

    print(
        f"\nSplit thresholds: train_thr={train_thr:,}, "
        f"val_thr={val_thr:,}"
    )
    return train, val, test


def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}")


def print_split_summary(name: str, part: pd.DataFrame) -> None:
    if len(part) == 0:
        print(f"  {name:>5}: EMPTY")
        return

    n_pos = int(part["label"].sum())
    pos_rate = n_pos / len(part)
    ts_min = int(part["timestamp"].min())
    ts_max = int(part["timestamp"].max())

    print(
        f"  {name:>5}: rows={len(part):>8,}, "
        f"pos_rate={pos_rate:>6.2%}, "
        f"users={part['uid'].nunique():>3}, "
        f"items={part['item_id'].nunique():>6,}, "
        f"ts=[{ts_min:,} .. {ts_max:,}]"
    )


def check_temporal_order(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Убеждаемся, что все события train строго раньше val, а val — раньше test."""
    if len(train) > 0 and len(val) > 0:
        if int(train["timestamp"].max()) >= int(val["timestamp"].min()):
            raise RuntimeError(
                "Temporal order broken: train contains events "
                "not strictly before val."
            )
    if len(val) > 0 and len(test) > 0:
        if int(val["timestamp"].max()) >= int(test["timestamp"].min()):
            raise RuntimeError(
                "Temporal order broken: val contains events "
                "not strictly before test."
            )
    print("  temporal order: OK (train < val < test)")


def check_overlaps(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Смотрим, насколько пользователи пересекаются между train и test."""
    train_users = set(train["uid"].unique())
    test_users = set(test["uid"].unique())

    if not test_users:
        _warn("test split has no users")
        return

    overlap = train_users & test_users
    cold_start = test_users - train_users

    print(
        f"  user overlap train/test: "
        f"{len(overlap)} of {len(test_users)} test users "
        f"(cold-start: {len(cold_start)})"
    )

    if cold_start:
        _warn(
            f"{len(cold_start)} test users are not present in train — "
            "ranking metrics for them are driven by priors only "
            "(all history features are zero)"
        )


def check_pos_rate_drift(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Предупреждаем, если positive rate в test сильно отклонился от train."""
    train_rate = float(train["label"].mean()) if len(train) > 0 else float("nan")
    val_rate = float(val["label"].mean()) if len(val) > 0 else float("nan")
    test_rate = float(test["label"].mean()) if len(test) > 0 else float("nan")

    for name, rate in (("val", val_rate), ("test", test_rate)):
        if pd.isna(rate):
            continue
        drift = abs(rate - train_rate)
        if drift > POS_RATE_DRIFT_WARN:
            _warn(
                f"pos_rate drift train->{name}: "
                f"{train_rate:.2%} -> {rate:.2%} "
                f"(Δ={drift:.2%}, threshold={POS_RATE_DRIFT_WARN:.2%})"
            )


def check_test_size(test: pd.DataFrame) -> None:
    """Предупреждаем, если в test слишком мало уникальных пользователей."""
    n_users = test["uid"].nunique()
    if n_users < MIN_TEST_USERS_WARN:
        _warn(
            f"only {n_users} unique users in test "
            f"(threshold={MIN_TEST_USERS_WARN}); "
            "ranking metrics will be noisy"
        )


def check_cold_start_share(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Считает, какую долю test-строк дают пользователи без истории в train.

    Для этих строк все u_* и p_* признаки равны 0, и модель предсказывает
    только по i_* и ctx_*. Чем больше доля, тем сильнее метрики test
    измеряют cold-start качество, а не качество на «тёплых» пользователях.
    """
    train_users = set(train["uid"].unique())
    cold_mask = ~test["uid"].isin(train_users)
    n_cold_rows = int(cold_mask.sum())

    if n_cold_rows == 0:
        return

    share = n_cold_rows / len(test)
    print(
        f"  cold-start rows in test: {n_cold_rows:,} "
        f"({share:.2%} of test rows)"
    )

    if share > 0.20:
        _warn(
            f"cold-start accounts for {share:.2%} of test rows — "
            "test metrics will be dominated by cold-start behaviour; "
            "discuss this in README"
        )


def main() -> None:
    features = load_features(FEATURES_PATH)
    print(f"Loaded features: {features.shape}")

    train, val, test = split_by_time(features, TRAIN_FRAC, VAL_FRAC)

    print("\n=== Split summary ===")
    print_split_summary("train", train)
    print_split_summary("val", val)
    print_split_summary("test", test)

    print("\n=== Sanity checks ===")
    check_temporal_order(train, val, test)
    check_overlaps(train, val, test)
    check_pos_rate_drift(train, val, test)
    check_test_size(test)
    check_cold_start_share(train, test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUTPUT_DIR / "train.parquet", index=False)
    val.to_parquet(OUTPUT_DIR / "val.parquet", index=False)
    test.to_parquet(OUTPUT_DIR / "test.parquet", index=False)

    print(
        f"\nSaved to {OUTPUT_DIR}/: "
        f"train.parquet, val.parquet, test.parquet"
    )
    print("--> split_data done")


if __name__ == "__main__":
    main()
