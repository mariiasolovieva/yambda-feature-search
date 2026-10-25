from datasets import load_dataset

dataset = load_dataset(
    "yandex/yambda",
    data_dir="flat/50m",
    data_files="multi_event.parquet",
    split="train",
    streaming=True,
)

for i, row in enumerate(dataset):
    print(row)
    if i == 4:
        break
    