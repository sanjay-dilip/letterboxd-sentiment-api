from typing import Dict, Any, List, Optional
import pandas as pd
from .config import CLEANED_DATA_PATH
from .sentiment_model import load_model_and_vectorizer, predict_sentiment_for_text

class MovieAnalytics:
    """
    Analytics layer over the cleaned Letterboxd review dataset.
    - Loads cleaned_reviews.csv
    - Optionally computes a sentiment_score for each review
    - Provides helpers to summarize movies
    """

    def __init__(
        self,
        movie_col: str = "movie_title",
        text_col: str = "clean_text",
        label_col: str = "sentiment_label",
        rating_col: str = "rating_numeric",
    ) -> None:
        self.movie_col = movie_col
        self.text_col = text_col
        self.label_col = label_col
        self.rating_col = rating_col
        # Load cleaned data
        self.df = pd.read_csv(CLEANED_DATA_PATH)
        # Load model + vectorizer once
        self.vectorizer, self.model = load_model_and_vectorizer()
        # Ensure we have a numeric sentiment_score column
        if "sentiment_score" not in self.df.columns:
            self._add_sentiment_scores()

    def _add_sentiment_scores(self) -> None:
        """
        Add a sentiment_score column for each review.
        Here we'll define sentiment_score as:
        - probability that the review is "positive"
        according to our model.
        """
        scores: List[float] = []
        for text in self.df[self.text_col].astype(str):
            result = predict_sentiment_for_text(text, self.vectorizer, self.model)
            # probability for the "positive" class, fall back if not present
            all_probs = result.get("all_probs", {})
            score = all_probs.get("positive", 0.0)
            scores.append(float(score))
        self.df["sentiment_score"] = scores

    def get_available_movies(self, min_reviews: int = 1) -> List[str]:
        """
        Return a sorted list of movie titles that have at least min_reviews.
        """
        counts = self.df[self.movie_col].value_counts()
        valid = counts[counts >= min_reviews].index.tolist()
        return sorted(valid)

    def get_movie_summary(self, movie: str) -> Dict[str, Any]:
        """
        Summarize sentiment and rating stats for a single movie.
        Returns a dict with:
        - movie title
        - number of reviews
        - avg numeric rating (Letterboxd stars)
        - avg sentiment_score (model)
        - fraction of pos / neu / neg labels
        """
        subset = self.df[self.df[self.movie_col] == movie]
        if subset.empty:
            return {
                "movie": movie,
                "count": 0,
                "message": "No reviews found for this movie.",
            }
        count = int(len(subset))
        # average star rating if available
        avg_rating = None
        if self.rating_col in subset.columns:
            avg_rating = float(subset[self.rating_col].dropna().mean())
        # model-based sentiment score
        avg_sentiment_score = float(subset["sentiment_score"].mean())
        # label distribution
        label_counts = subset[self.label_col].value_counts(normalize=True)
        pos_pct = float(label_counts.get("positive", 0.0))
        neg_pct = float(label_counts.get("negative", 0.0))
        neu_pct = float(label_counts.get("neutral", 0.0))
        return {
            "movie": movie,
            "count": count,
            "avg_rating": avg_rating,
            "avg_sentiment_score": avg_sentiment_score,
            "positive_pct": pos_pct,
            "neutral_pct": neu_pct,
            "negative_pct": neg_pct,
        }

    def compare_movies(self, movie_a: str, movie_b: str) -> Dict[str, Any]:
        """
        Compare two movies side by side.
        """
        summary_a = self.get_movie_summary(movie_a)
        summary_b = self.get_movie_summary(movie_b)
        return {"movie_a": summary_a, "movie_b": summary_b}

    def top_movies_by_sentiment(
        self,
        min_reviews: int = 10,
        top_n: int = 10,
        direction: str = "desc",
    ) -> pd.DataFrame:
        """
        Return a DataFrame of top movies ranked by avg_sentiment_score.
        direction: "desc" for highest sentiment, "asc" for lowest.
        """
        # group by movie
        grouped = (
            self.df.groupby(self.movie_col)
            .agg(
                count=("clean_text", "size"),
                avg_rating=(self.rating_col, "mean"),
                avg_sentiment_score=("sentiment_score", "mean"),
            )
            .reset_index()
            .rename(columns={self.movie_col: "movie"})
        )
        # filter by minimum reviews
        grouped = grouped[grouped["count"] >= min_reviews]
        # sort
        ascending = direction == "asc"
        grouped = grouped.sort_values("avg_sentiment_score", ascending=ascending)
        # limit
        return grouped.head(top_n)