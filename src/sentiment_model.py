from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from .config import CLEANED_DATA_PATH, MODELS_DIR


def load_training_data(
    text_col: str = "clean_text",
    label_col: str = "sentiment_label_strict",
    min_len: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cleaned_reviews.csv, filter text, and balance classes
    using strict binary labels.

    Only rows where sentiment_label_strict is 'positive' or 'negative'
    are used. Others are ignored.
    """
    df = pd.read_csv(CLEANED_DATA_PATH)

    # Drop missing text or labels
    df = df.dropna(subset=[text_col, label_col])

    # Keep only strict positive and negative
    df = df[df[label_col].isin(["positive", "negative"])]

    # Filter short reviews more aggressively
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.len() >= min_len]

    # Split by class
    df_pos = df[df[label_col] == "positive"]
    df_neg = df[df[label_col] == "negative"]

    if len(df_pos) == 0 or len(df_neg) == 0:
        print("Warning: one of the classes is empty after filtering.")
        X = df[text_col].values
        y = df[label_col].values
        return X, y

    # Balance classes by downsampling the majority
    min_size = min(len(df_pos), len(df_neg))

    df_pos = resample(df_pos, replace=False, n_samples=min_size, random_state=42)
    df_neg = resample(df_neg, replace=False, n_samples=min_size, random_state=42)

    # Combine and shuffle
    df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42)

    X = df_balanced[text_col].values
    y = df_balanced[label_col].values

    print("Training label counts (balanced):")
    print(df_balanced[label_col].value_counts())

    return X, y


def train_sentiment_model(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Train TF IDF plus Logistic Regression on cleaned_reviews.csv
    using strict binary labels.
    """
    X, y = load_training_data()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        min_df=5,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_val_vec)
    print("Validation results (binary strict labels):")
    print(classification_report(y_val, y_pred))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "sentiment_model.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "vectorizer.joblib")

    return vectorizer, model


def load_model_and_vectorizer() -> Tuple[TfidfVectorizer, LogisticRegression]:
    """
    Load the baseline TF IDF and Logistic Regression model
    from the models folder.
    """
    vectorizer: TfidfVectorizer = joblib.load(MODELS_DIR / "vectorizer.joblib")
    model: LogisticRegression = joblib.load(MODELS_DIR / "sentiment_model.joblib")
    return vectorizer, model


def predict_sentiment_for_text(
    text: str,
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
) -> dict:
    """
    Run sentiment prediction for a single raw text string.

    Returns:
        {
          "label": "positive" or "negative",
          "probability": float,
          "all_probs": {label: prob, ...}
        }
    """
    if not isinstance(text, str):
        text = ""

    X_vec = vectorizer.transform([text])
    probs = model.predict_proba(X_vec)[0]
    labels = model.classes_

    best_idx = int(np.argmax(probs))

    return {
        "label": str(labels[best_idx]),
        "probability": float(probs[best_idx]),
        "all_probs": {str(lbl): float(p) for lbl, p in zip(labels, probs)},
    }


if __name__ == "__main__":
    # simple cli entry point: python -m src.sentiment_model
    train_sentiment_model()