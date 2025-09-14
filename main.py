import pandas as pd
import bz2
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")
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

base_path = "/Users/inban/.cache/kagglehub/datasets/bittlingmayer/amazonreviews/versions/7"

train_df = load_amazon_data(f"{base_path}/train.ft.txt.bz2", nrows=10000)
test_df = load_amazon_data(f"{base_path}/test.ft.txt.bz2", nrows=1000)


# Keep both for now, so we can route complaints (label 1) and tag praise (label 2)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

def classify_negative_review(text):
    prompt = (
        "You are a customer service AI assistant. Read the following customer complaint and categorize it into one of three types:\n"
        "- Delivery Issue: Problems related to shipping delays, damaged packaging, missing items, or incorrect delivery.\n"
        "- Product Issue: Problems with the functionality, quality, description mismatch, or defects in the item.\n"
        "- Other: Anything not related to delivery or the product itself (e.g., customer service, pricing, returns).\n"
        "Only return the name of the category—no explanation.\n\n"
        f"Customer Review: {text}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()

# Apply classifier to first 10 rows for testing

negative_reviews = train_df[train_df["label"] == 1].copy()
negative_reviews = negative_reviews.reset_index(drop=True)

positive_reviews = train_df[train_df["label"] == 2].copy()

sample = negative_reviews.head(10).copy()
sample["category"] = sample["text"].apply(classify_negative_review)

print(sample[["text", "category"]])
