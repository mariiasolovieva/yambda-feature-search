from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path("data/events_sample.parquet")
OUTPUT_PATH = Path("results/features.parquet")

SECONDS_PER_TIMESTAMP_UNIT = 5
SECONDS_PER_DAY = 86_400

WINDOWS_DAYS = (7, 30)

TARGET_THRESHOLD_PCT = 50.0

DROP_ZERO_RATIO = True

ALL_TIME_UNITS = int(50 * 365 * SECONDS_PER_DAY / SECONDS_PER_TIMESTAMP_UNIT)

# load events

def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    events = pd.read_parquet(path)

    required = {
        "uid", "item_id", "timestamp", "event_type",
        "is_organic", "played_ratio_pct", "track_length_seconds",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return events


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    """Add helper-columns and checks invariants"""
    events = events.copy()
    events["timestamp"] = events["timestamp"].astype("int64")

    keep_types = ["listen", "like", "unlike", "dislike", "undislike"]
    events = events[events["event_type"].isin(keep_types)].copy()

    events["one"] = 1.0
    events["is_listen"] = (events["event_type"] == "listen").astype("int8")
    events["is_like"] = (events["event_type"] == "like").astype("int8")
    events["is_unlike"] = (events["event_type"] == "unlike").astype("int8")
    events["is_dislike"] = (events["event_type"] == "dislike").astype("int8")
    events["played_ratio_pct"] = events["played_ratio_pct"].fillna(0.0)
    events["played_ratio_x_listen"] = (
        events["played_ratio_pct"] * events["is_listen"]
    )

    bad = events.loc[
        (events["event_type"] != "listen") & (events["played_ratio_pct"] > 0)
    ]
    if len(bad) > 0:
        raise ValueError(
            f"played_ratio_pct > 0 on non-listen events: {len(bad)}"
        )

    return events


def _window_units(days: float) -> int:
    return int(days * SECONDS_PER_DAY / SECONDS_PER_TIMESTAMP_UNIT)


def _window_sum(
    targets: pd.DataFrame,
    source: pd.DataFrame,
    group_cols: list[str],
    window_units: int,
    value_col: str,
) -> np.ndarray:
    """
    The sum of the `value_col` for source events with the same group_cols and 
    timestamp is strictly in the interval (t - window_units, t).
    Returns an array aligned in the order of `targets`
    """
    src = source[group_cols + ["timestamp", value_col]].copy()
    src = src.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
    src["_cum"] = src.groupby(group_cols, sort=False)[value_col].cumsum()

    tgt = targets[group_cols + ["timestamp"]].copy()
    tgt["_orig_idx"] = np.arange(len(tgt))
    tgt = tgt.sort_values(["timestamp"], kind="stable").reset_index(drop=True)

    # cumulative by the moment t - 1
    q_t = tgt.copy()
    q_t["timestamp"] = q_t["timestamp"] - 1
    m_t = pd.merge_asof(
        q_t,
        src[group_cols + ["timestamp", "_cum"]],
        on="timestamp",
        by=group_cols,
        direction="backward",
    )
    cum_t = m_t["_cum"].fillna(0.0).to_numpy()

    # cumulative by the moment t - window - 1
    q_tw = tgt.copy()
    q_tw["timestamp"] = q_tw["timestamp"] - window_units - 1
    m_tw = pd.merge_asof(
        q_tw,
        src[group_cols + ["timestamp", "_cum"]],
        on="timestamp",
        by=group_cols,
        direction="backward",
    )
    cum_tw = m_tw["_cum"].fillna(0.0).to_numpy()

    result = np.zeros(len(targets), dtype="float64")
    result[m_t["_orig_idx"].to_numpy()] = cum_t - cum_tw
    return result


def _recency_days(
    targets: pd.DataFrame,
    source: pd.DataFrame,
    group_cols: list[str],
) -> np.ndarray:
    """How many days have passed since the last source event < t. NaN if there is no history"""
    src = source[group_cols + ["timestamp"]].copy()
    src = src.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
    src = src.rename(columns={"timestamp": "src_ts"})

    tgt = targets[group_cols + ["timestamp"]].copy()
    tgt["_orig_idx"] = np.arange(len(tgt))
    tgt["query_ts"] = tgt["timestamp"] - 1
    tgt = tgt.sort_values(["query_ts"], kind="stable").reset_index(drop=True)

    m = pd.merge_asof(
        tgt,
        src,
        left_on="query_ts",
        right_on="src_ts",
        by=group_cols,
        direction="backward",
    )

    delta_units = (m["query_ts"] - m["src_ts"]).to_numpy()
    delta_days = delta_units * SECONDS_PER_TIMESTAMP_UNIT / SECONDS_PER_DAY

    result = np.full(len(targets), np.nan, dtype="float64")
    result[m["_orig_idx"].to_numpy()] = delta_days
    return result


def _tenure_days(
    targets: pd.DataFrame,
    source: pd.DataFrame,
    group_cols: list[str],
) -> np.ndarray:
    """How many days have passed since the first source event, NaN if there is no history"""
    src = source[group_cols + ["timestamp"]].copy()
    src = src.sort_values(["timestamp"], kind="stable").reset_index(drop=True)
    src["_min"] = src.groupby(group_cols, sort=False)["timestamp"].cummin()
    src = src.rename(columns={"timestamp": "src_ts"})

    tgt = targets[group_cols + ["timestamp"]].copy()
    tgt["_orig_idx"] = np.arange(len(tgt))
    tgt["query_ts"] = tgt["timestamp"] - 1
    tgt = tgt.sort_values(["query_ts"], kind="stable").reset_index(drop=True)

    m = pd.merge_asof(
        tgt,
        src[group_cols + ["src_ts", "_min"]],
        left_on="query_ts",
        right_on="src_ts",
        by=group_cols,
        direction="backward",
    )

    delta_units = (m["query_ts"] - m["_min"]).to_numpy()
    delta_days = delta_units * SECONDS_PER_TIMESTAMP_UNIT / SECONDS_PER_DAY

    result = np.full(len(targets), np.nan, dtype="float64")
    result[m["_orig_idx"].to_numpy()] = delta_days
    return result


def _safe_ratio(num, den) -> np.ndarray:
    """handle the case with den == 0."""
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den > 0)
    return out


# feature groups

def build_user_features(
    listens: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """user's features for all his events are up to t"""
    result = pd.DataFrame(index=listens.index)
    group = ["uid"]

    # All-time
    result["u_n_events"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "one"
    )
    result["u_n_listens"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_listen"
    )
    result["u_n_likes"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_like"
    )
    result["u_n_dislikes"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_dislike"
    )

    sum_pr_all = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "played_ratio_x_listen"
    )
    result["u_like_rate"] = _safe_ratio(
        result["u_n_likes"], result["u_n_listens"]
    )
    result["u_mean_played_ratio"] = _safe_ratio(
        sum_pr_all, result["u_n_listens"]
    )

    # window-based
    for wd in WINDOWS_DAYS:
        wu = _window_units(wd)
        n_listens = _window_sum(listens, events, group, wu, "is_listen")
        n_likes = _window_sum(listens, events, group, wu, "is_like")
        sum_pr = _window_sum(
            listens, events, group, wu, "played_ratio_x_listen"
        )

        result[f"u_n_listens_{wd}d"] = n_listens
        result[f"u_n_likes_{wd}d"] = n_likes
        result[f"u_mean_played_ratio_{wd}d"] = _safe_ratio(
            sum_pr, n_listens
        )
        result[f"u_like_rate_{wd}d"] = _safe_ratio(n_likes, n_listens)

    # Recency / tenure
    result["u_days_since_last_event"] = _recency_days(
        listens, events, group
    )
    result["u_days_since_first_event"] = _tenure_days(
        listens, events, group
    )

    return result


def build_item_features(
    listens: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Track signs for all events with this track are up to t"""
    result = pd.DataFrame(index=listens.index)
    group = ["item_id"]

    result["i_n_events"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "one"
    )
    result["i_n_listens"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_listen"
    )
    result["i_n_likes"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_like"
    )
    result["i_n_dislikes"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_dislike"
    )

    sum_pr_all = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "played_ratio_x_listen"
    )
    result["i_like_rate"] = _safe_ratio(
        result["i_n_likes"], result["i_n_listens"]
    )
    result["i_mean_played_ratio"] = _safe_ratio(
        sum_pr_all, result["i_n_listens"]
    )

    for wd in WINDOWS_DAYS:
        wu = _window_units(wd)
        n_listens = _window_sum(listens, events, group, wu, "is_listen")
        n_likes = _window_sum(listens, events, group, wu, "is_like")
        sum_pr = _window_sum(
            listens, events, group, wu, "played_ratio_x_listen"
        )

        result[f"i_n_listens_{wd}d"] = n_listens
        result[f"i_n_likes_{wd}d"] = n_likes
        result[f"i_mean_played_ratio_{wd}d"] = _safe_ratio(
            sum_pr, n_listens
        )

    # no likes without listens
    listens_only = events[events["event_type"] == "listen"]
    result["i_days_since_last_listen"] = _recency_days(
        listens, listens_only, group
    )
    result["i_days_since_first_listen"] = _tenure_days(
        listens, listens_only, group
    )

    return result


def build_pair_features(
    listens: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """The attributes of a user–item pair based on their joint history are up to t"""
    result = pd.DataFrame(index=listens.index)
    group = ["uid", "item_id"]

    result["p_n_events"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "one"
    )
    result["p_n_listens"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_listen"
    )
    result["p_n_likes"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_like"
    )
    result["p_n_unlikes"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_unlike"
    )
    result["p_n_dislikes"] = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "is_dislike"
    )

    sum_pr_all = _window_sum(
        listens, events, group, ALL_TIME_UNITS, "played_ratio_x_listen"
    )
    result["p_mean_played_ratio"] = _safe_ratio(
        sum_pr_all, result["p_n_listens"]
    )

    result["p_user_has_liked_item"] = (
        (result["p_n_likes"] > result["p_n_unlikes"]).astype("int8")
    )

    for wd in WINDOWS_DAYS:
        wu = _window_units(wd)
        result[f"p_n_listens_{wd}d"] = _window_sum(
            listens, events, group, wu, "is_listen"
        )

    listens_only = events[events["event_type"] == "listen"]
    result["p_days_since_last_listen"] = _recency_days(
        listens, listens_only, group
    )

    return result


def build_context_features(listens: pd.DataFrame) -> pd.DataFrame:
    """Signs that are known at the time of the start of listening (without history)"""
    result = pd.DataFrame(index=listens.index)

    result["ctx_track_length_sec"] = (
        listens["track_length_seconds"].astype("float64")
    )
    result["ctx_log_track_length"] = np.log1p(
        result["ctx_track_length_sec"].clip(lower=0)
    )

    result["ctx_is_organic"] = (
        listens["is_organic"].fillna(-1).astype("int8")
    )

    return result


def build_dataset(
    listens: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:

    print("Building user features...")
    user_feats = build_user_features(listens, events)
    print(f"  -> {user_feats.shape[1]} user features")

    print("Building item features...")
    item_feats = build_item_features(listens, events)
    print(f"  -> {item_feats.shape[1]} item features")

    print("Building pair features...")
    pair_feats = build_pair_features(listens, events)
    print(f"  -> {pair_feats.shape[1]} pair features")

    print("Building context features...")
    ctx_feats = build_context_features(listens)
    print(f"  -> {ctx_feats.shape[1]} context features")

    base = listens[
        ["uid", "item_id", "timestamp", "played_ratio_pct", "label"]
    ].copy()
    result = pd.concat(
        [base, user_feats, item_feats, pair_feats, ctx_feats],
        axis=1,
    )
    return result


def main() -> None:
    events = load_events(INPUT_PATH)
    events = prepare_events(events)

    print(f"Events after filtering: {len(events):,}")
    print(f"Unique users: {events['uid'].nunique():,}")
    print(f"Unique items: {events['item_id'].nunique():,}")

    listens = events[events["event_type"] == "listen"].copy()
    print(f"\nListen events: {len(listens):,}")

    if DROP_ZERO_RATIO:
        n_before = len(listens)
        listens = listens[listens["played_ratio_pct"] > 0].copy()
        print(
            f"Drop played_ratio_pct == 0: "
            f"{n_before - len(listens):,} rows removed"
        )

    listens["label"] = (
        listens["played_ratio_pct"] >= TARGET_THRESHOLD_PCT
    ).astype("int8")

    pos_rate = listens["label"].mean()
    print(
        f"\nTarget: played_ratio_pct >= {TARGET_THRESHOLD_PCT:.0f}  "
        f"(positive rate = {pos_rate:.2%})"
    )

    features = build_dataset(listens, events)
    print(f"\nFull feature matrix: {features.shape}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print("--> build_features done")


if __name__ == "__main__":
    main()
