from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/events_sample.parquet")

# В вашем analyze_like_delays.py стоит 5 — оставляем как есть,
# но при необходимости проверьте по документации Yambda.
SECONDS_PER_TIMESTAMP_UNIT = 5

# Кандидатные пороги для бинаризации played_ratio_pct
THRESHOLDS_PCT = [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99, 100]

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

# Окно, в пределах которого лайк считается "ответом" на прослушивание
MAX_LIKE_LOOKBACK_HOURS = 7 * 24


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    events = pd.read_parquet(path)

    required = {"uid", "item_id", "timestamp", "event_type", "played_ratio_pct"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return events


def print_basic_stats(events: pd.DataFrame) -> pd.DataFrame:
    print(f"All events: {len(events):,}")
    print(f"All users: {events['uid'].nunique():,}")
    print(f"All items: {events['item_id'].nunique():,}")

    print("\nEvent type distribution:")
    counts = events["event_type"].value_counts()
    for et, cnt in counts.items():
        print(f"  {et:>10}: {cnt:>10,} ({cnt / len(events) * 100:>6.2f}%)")

    non_listen = events[events["event_type"] != "listen"]
    n_non_listen_with_ratio = int(non_listen["played_ratio_pct"].notna().sum())
    if n_non_listen_with_ratio:
        print(
            "\n[WARNING] played_ratio_pct is non-null for non-listen events: "
            f"{n_non_listen_with_ratio:,} rows"
        )
    else:
        print("\nOK: played_ratio_pct is defined only for listen events")

    listens = events[events["event_type"] == "listen"].copy()
    print(f"\nListen events: {len(listens):,}")

    n_nan = int(listens["played_ratio_pct"].isna().sum())
    if n_nan:
        print(f"NaN in played_ratio_pct among listens: {n_nan:,}")

    return listens


def print_distribution(listens: pd.DataFrame) -> pd.Series:
    ratios = listens["played_ratio_pct"].dropna()

    print("\n=== played_ratio_pct distribution ===")
    print(f"n       : {len(ratios):,}")
    print(f"min     : {ratios.min():.2f}")
    print(f"max     : {ratios.max():.2f}")
    print(f"mean    : {ratios.mean():.2f}")
    print(f"median  : {ratios.median():.2f}")
    print(f"std     : {ratios.std():.2f}")

    n_zero = int((ratios == 0).sum())
    n_over_100 = int((ratios > 100).sum())
    print(
        f"\nzeros (skip right away)   : {n_zero:,} "
        f"({n_zero / len(ratios) * 100:.2f}%)"
    )
    print(
        f"values > 100% (rewinds)   : {n_over_100:,} "
        f"({n_over_100 / len(ratios) * 100:.2f}%)"
    )

    print("\nQuantiles:")
    print(ratios.quantile(QUANTILES).round(2).to_string())

    return ratios


def print_threshold_table(ratios: pd.Series) -> None:
    print("\n=== Class balance at candidate thresholds ===")
    header = (
        f"{'threshold':>10} | {'positives':>12} | {'negatives':>12} | "
        f"{'pos rate':>9} | {'neg/pos':>9}"
    )
    print(header)
    print("-" * len(header))

    for t in THRESHOLDS_PCT:
        pos = int((ratios >= t).sum())
        neg = int(len(ratios) - pos)
        pos_rate = pos / len(ratios) if len(ratios) > 0 else 0.0
        neg_pos = neg / pos if pos > 0 else float("inf")
        print(
            f"{t:>9}% | {pos:>12,} | {neg:>12,} | "
            f"{pos_rate:>8.2%} | 1:{neg_pos:>8.2f}"
        )


def print_histogram(ratios: pd.Series) -> None:
    print("\n=== Histogram (bin = 10%) ===")
    bins = np.arange(0, 111, 10)
    binned = pd.cut(ratios, bins=bins, include_lowest=True, right=False)
    counts = binned.value_counts().sort_index()

    total = len(ratios)
    max_count = int(counts.max()) if len(counts) > 0 else 0

    for interval, cnt in counts.items():
        cnt = int(cnt)
        pct = cnt / total * 100 if total else 0.0
        bar_len = int(cnt / max_count * 50) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"{str(interval):>18}: {cnt:>9,} ({pct:>5.2f}%) {bar}")


def print_organic_breakdown(listens: pd.DataFrame) -> None:
    if "is_organic" not in listens.columns:
        return

    print("\n=== Distribution by is_organic ===")
    for val, grp in listens.groupby("is_organic", dropna=False):
        r = grp["played_ratio_pct"].dropna()
        if len(r) == 0:
            continue
        print(f"\n  is_organic = {val}  (n = {len(r):,})")
        print(f"    mean={r.mean():.2f}, median={r.median():.2f}")
        for t in (50, 80, 90):
            share = (r >= t).mean() * 100
            print(f"    share >= {t}%: {share:.2f}%")


def analyze_like_correlation(
    listens: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    likes = events[events["event_type"] == "like"].copy()
    if len(likes) == 0:
        print("\n[skip] No like events found")
        return

    print("\n=== Correlation between played_ratio_pct and likes ===")
    print(f"Total like events: {len(likes):,}")

    tolerance_units = int(
        MAX_LIKE_LOOKBACK_HOURS * 3600 / SECONDS_PER_TIMESTAMP_UNIT
    )

    # Только listen-события с определённым played_ratio_pct
    listens_clean = listens[listens["played_ratio_pct"].notna()][
        ["uid", "item_id", "timestamp", "played_ratio_pct"]
    ].copy()
    listens_clean = listens_clean.sort_values(
        ["timestamp", "uid", "item_id"], kind="stable"
    ).reset_index(drop=True)

    likes_sorted = likes[["uid", "item_id", "timestamp"]].copy()
    likes_sorted = likes_sorted.sort_values(
        ["timestamp", "uid", "item_id"], kind="stable"
    ).reset_index(drop=True)

    # ---- Часть A: для каждого лайка найти предшествующий listen ----
    listens_for_a = listens_clean.copy()
    listens_for_a["listen_ts"] = listens_for_a["timestamp"]

    matched_a = pd.merge_asof(
        likes_sorted,
        listens_for_a,
        on="timestamp",
        by=["uid", "item_id"],
        direction="backward",
        tolerance=tolerance_units,
    )
    # В matched_a: "timestamp" = время лайка,
    # "played_ratio_pct" = played_ratio соответствующего прослушивания
    # "listen_ts" = время этого прослушивания (NaN, если не нашли)

    n_likes = len(matched_a)
    n_with_listen = int(matched_a["played_ratio_pct"].notna().sum())
    print(
        f"\nLikes with a preceding listen within {MAX_LIKE_LOOKBACK_HOURS}h: "
        f"{n_with_listen:,} / {n_likes:,} "
        f"({n_with_listen / n_likes * 100:.2f}%)"
    )

    if n_with_listen == 0:
        return

    like_ratios = matched_a["played_ratio_pct"].dropna()

    print("\nplayed_ratio_pct of the listen preceding a like:")
    print(like_ratios.quantile(QUANTILES).round(2).to_string())
    print(
        f"\nmean={like_ratios.mean():.2f}, median={like_ratios.median():.2f}"
    )

    print("\nShare of 'like-preceding' listens above threshold:")
    for t in THRESHOLDS_PCT:
        share = (like_ratios >= t).mean() * 100
        print(f"  >= {t:>3}%: {share:>6.2f}%")

    # ---- Часть B: для каждого listen проверить, был ли лайк после ----
    likes_for_b = likes_sorted.copy()
    likes_for_b["like_ts"] = likes_for_b["timestamp"]

    matched_b = pd.merge_asof(
        listens_clean,
        likes_for_b,
        on="timestamp",
        by=["uid", "item_id"],
        direction="forward",
        tolerance=tolerance_units,
    )
    # В matched_b: "like_ts" = время ближайшего лайка после listen
    # (NaN, если в пределах tolerance лайка не было)

    matched_b["followed_by_like"] = matched_b["like_ts"].notna()
    n_followed = int(matched_b["followed_by_like"].sum())
    base_rate = n_followed / len(matched_b) if len(matched_b) > 0 else 0.0

    print(
        f"\nListens followed by a like within "
        f"{MAX_LIKE_LOOKBACK_HOURS}h: {n_followed:,} / "
        f"{len(matched_b):,} (base rate = {base_rate * 100:.4f}%)"
    )

    if n_followed == 0:
        return

    print(
        "\nQuality of the threshold as a like-predictor:\n"
        f"{'T':>4} | {'pred_pos':>10} | {'TP':>8} | "
        f"{'precision':>10} | {'recall':>8} | {'F1':>7} | {'lift':>6}"
    )
    print("-" * 80)

    for t in THRESHOLDS_PCT:
        pred_pos = matched_b["played_ratio_pct"] >= t
        n_pred_pos = int(pred_pos.sum())
        if n_pred_pos == 0:
            continue
        tp = int((pred_pos & matched_b["followed_by_like"]).sum())
        precision = tp / n_pred_pos
        recall = tp / n_followed if n_followed > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        lift = precision / base_rate if base_rate > 0 else float("inf")
        print(
            f"{t:>3}% | {n_pred_pos:>10,} | {tp:>8,} | "
            f"{precision:>10.4f} | {recall:>8.4f} | "
            f"{f1:>7.4f} | {lift:>5.1f}x"
        )


def main():
    events = load_events(INPUT_PATH)
    listens = print_basic_stats(events)
    ratios = print_distribution(listens)
    print_threshold_table(ratios)
    print_histogram(ratios)
    print_organic_breakdown(listens)
    analyze_like_correlation(listens, events)
    print("\n--> Done.")


if __name__ == "__main__":
    main()
    