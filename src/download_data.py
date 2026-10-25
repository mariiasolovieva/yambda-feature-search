
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# data source
DATASET_NAME = "yandex/yambda"
DATA_DIR = "flat/50m"
DATA_FILE = "multi_event.parquet"

# set parameters
N_USERS = 100
RANDOM_SEED = 42

OUTPUT_PATH = Path("data/events_sample.parquet")


def load_yambda():
    """open stream reading of a dataset"""
    return load_dataset(
        DATASET_NAME,
        data_dir=DATA_DIR,
        data_files=DATA_FILE,
        split="train",
        streaming=True,
    )


def main():
    print("I. Collect user list")

    dataset = load_yambda()
    unique_users = []
    previous_uid = None

    for row in dataset:
        uid = int(row["uid"])

        if uid != previous_uid:
            unique_users.append(uid)
            previous_uid = uid

    print(f"--> Unique users: {len(unique_users):,}")

    if len(unique_users) < N_USERS:
        raise RuntimeError(
            f"dataset has less than {N_USERS}."
        )

    # random choice of 100 users, fixed random_seed
    rng = random.Random(RANDOM_SEED)
    selected_users = set(rng.sample(unique_users, N_USERS))
    last_selected_uid = max(selected_users)

    print(f"--> number of unique users chosen: {len(selected_users)}")
    print("II. Collect events for the randomly chosen users")

    dataset = load_yambda()
    rows = []

    for row in dataset:
        uid = int(row["uid"])

        if uid > last_selected_uid:
            break

        if uid not in selected_users:
            continue

        rows.append({
            "uid": uid,
            "item_id": int(row["item_id"]),
            "timestamp": int(row["timestamp"]),
            "event_type": str(row["event_type"]),
            "is_organic": row["is_organic"],
            "played_ratio_pct": row["played_ratio_pct"],
            "track_length_seconds": row["track_length_seconds"],
        })

    if not rows:
        raise RuntimeError("Failed to collect users events")

    events = pd.DataFrame(rows)
    events = events.sort_values(
        ["uid", "timestamp"]
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(OUTPUT_PATH, index=False)

    print("\n--> Download done.")
    print(f"File: {OUTPUT_PATH}")
    print(f"Events: {len(events):,}")
    print(f"Users: {events['uid'].nunique():,}")
    print(f"Unique tracks: {events['item_id'].nunique():,}")
    print("\nEvent types:")
    print(events["event_type"].value_counts())
    print("\nFirst five rows:")
    print(events.head())


if __name__ == "__main__":
    main()
