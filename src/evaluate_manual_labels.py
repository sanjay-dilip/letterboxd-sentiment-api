# src/evaluate_manual_labels.py

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

from .config import MODELS_DIR
from .sentiment_model import load_model_and_vectorizer, predict_sentiment_for_text
from .transformer_model import load_transformer_model, predict_transformer_sentiment

MANUAL_EVAL_PATH = Path("data/processed/manual_eval_samples.csv")


def main() -> None:
    if not MANUAL_EVAL_PATH.exists():
        raise FileNotFoundError(
            f"{MANUAL_EVAL_PATH} not found. Run sample_for_manual_labels "
            f"and add manual_sentiment labels first."
        )

    df = pd.read_csv(MANUAL_EVAL_PATH)

    if "manual_sentiment" not in df.columns:
        raise ValueError(
            "manual_sentiment column not found. Please label the file first."
        )

    df = df.dropna(subset=["clean_text", "manual_sentiment"])
    texts = df["clean_text"].astype(str).tolist()
    y_true = df["manual_sentiment"].astype(str).tolist()

    print(f"Loaded {len(df)} manually labeled samples.")

    # Baseline
    vectorizer, baseline_model = load_model_and_vectorizer()
    y_pred_baseline = []
    for t in texts:
        res = predict_sentiment_for_text(t, vectorizer, baseline_model)
        y_pred_baseline.append(res["label"])

    print("\nBaseline TF-IDF + Logistic Regression:")
    print(classification_report(y_true, y_pred_baseline))

    # Transformer
    tokenizer, transformer_model = load_transformer_model()
    y_pred_transformer = []
    for t in texts:
        res = predict_transformer_sentiment(t, tokenizer, transformer_model)
        y_pred_transformer.append(res["label"])

    print("\nDistilBERT (after SST-2 + Letterboxd fine tuning, if applied):")
    print(classification_report(y_true, y_pred_transformer))


if __name__ == "__main__":
    main()