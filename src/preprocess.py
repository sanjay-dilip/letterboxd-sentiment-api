import re
import pandas as pd
from .config import RAW_LETTERBOXD_PATH, CLEANED_DATA_PATH


def load_raw_letterboxd() -> pd.DataFrame:
    return pd.read_csv(RAW_LETTERBOXD_PATH, encoding="latin1")


def fix_rating_string(s: str) -> float:
    """
    Convert the broken star strings to a numeric rating.

    Each 'â??' is a full star.
    'Â½' is a half star.
    """
    if not isinstance(s, str):
        return float("nan")

    stars = s.count("â??")
    half = "Â½" in s

    rating = float(stars)
    if half:
        rating += 0.5
    return rating


def rating_to_label_loose(rating: float) -> str:
    """
    Original 3 class mapping, kept for analytics if needed.

    >= 3.5  -> positive
    <= 2.5  -> negative
    else    -> neutral
    """
    if pd.isna(rating):
        return "neutral"
    if rating >= 3.5:
        return "positive"
    if rating <= 2.5:
        return "negative"
    return "neutral"


def rating_to_label_strict(rating: float) -> str | None:
    """
    Strict binary mapping for training.

    >= 4.0  -> positive
    <= 2.0  -> negative
    else    -> None (ignore for training)
    """
    if pd.isna(rating):
        return None
    if rating >= 4.0:
        return "positive"
    if rating <= 2.0:
        return "negative"
    return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_clean_dataset() -> pd.DataFrame:
    """
    Build the cleaned dataset.

    Includes:
      - rating_numeric
      - sentiment_label_loose  (3 class, for analytics)
      - sentiment_label_strict (binary, for training)
    """
    df = load_raw_letterboxd()

    df = df.rename(
        columns={
            "Movie name": "movie_title",
            "Release Year": "release_year",
            "Rating": "rating_raw",
            "Reviewer name": "reviewer_name",
            "Review date": "review_date",
            "Review": "review_text",
            "Comment count": "comment_count",
            "Like count": "like_count",
        }
    )

    # Drop rows without review text
    df = df.dropna(subset=["review_text"])

    # Ratings
    df["rating_numeric"] = df["rating_raw"].apply(fix_rating_string)

    # Loose 3 class label
    df["sentiment_label"] = df["rating_numeric"].apply(rating_to_label_loose)

    # Strict binary label for training
    df["sentiment_label_strict"] = df["rating_numeric"].apply(rating_to_label_strict)

    # Clean text
    df["clean_text"] = df["review_text"].apply(clean_text)

    # Parse date
    try:
        df["review_date_parsed"] = pd.to_datetime(df["review_date"], errors="coerce")
    except Exception:
        df["review_date_parsed"] = pd.NaT

    cols = [
        "movie_title",
        "release_year",
        "review_date",
        "review_date_parsed",
        "reviewer_name",
        "rating_raw",
        "rating_numeric",
        "sentiment_label",          # loose 3 class
        "sentiment_label_strict",   # strict binary
        "review_text",
        "clean_text",
        "comment_count",
        "like_count",
    ]
    df = df[cols]

    return df
def save_clean_dataset(df: pd.DataFrame) -> None:
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_DATA_PATH, index=False)