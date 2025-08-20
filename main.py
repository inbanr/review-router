import pandas as pd
import bz2

def load_amazon_data(file_path, nrows=None):
    with bz2.open(file_path, "rt") as f:
        lines = f.readlines() if nrows is None else [next(f) for _ in range(nrows)]

    labels = []
    texts = []
    for line in lines:
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            label, text = parts
            labels.append(int(label[-1]))  # get 1 or 2
            texts.append(text)

    return pd.DataFrame({"label": labels, "text": texts})

# Change path to where your files are
base_path = "/Users/inban/.cache/kagglehub/datasets/bittlingmayer/amazonreviews/versions/7"

train_df = load_amazon_data(f"{base_path}/train.ft.txt.bz2", nrows=10000)  # sample 10k rows
test_df = load_amazon_data(f"{base_path}/test.ft.txt.bz2", nrows=1000)

print(train_df.head())