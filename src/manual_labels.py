# src/sample_for_manual_labels.py

import pandas as pd
from pathlib import Path

from .config import CLEANED_DATA_PATH

OUTPUT_PATH = Path("data/processed/manual_eval_samples.csv")


def main(n_samples_per_class: int = 75) -> None:
    df = pd.read_csv(CLEANED_DATA_PATH)

    # Use strict labels for sampling
    df = df.dropna(subset=["clean_text", "sentiment_label_strict"])
    df = df[df["sentiment_label_strict"].isin(["positive", "negative"])]

    # Sample equal pos / neg
    df_pos = df[df["sentiment_label_strict"] == "positive"].sample(
        n=n_samples_per_class, random_state=42
    )
    df_neg = df[df["sentiment_label_strict"] == "negative"].sample(
        n=n_samples_per_class, random_state=42
    )

    df_sample = (
        pd.concat([df_pos, df_neg])
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    # Keep just what you need to label by hand
    columns_to_keep = [
        "movie_title",
        "rating_numeric",
        "sentiment_label_strict",
        "review_text",
        "clean_text",
    ]
    df_sample = df_sample[columns_to_keep]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved sample for manual labeling to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()