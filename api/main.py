# api/main.py

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.sentiment_model import load_model_and_vectorizer, predict_sentiment_for_text
from src.analytics import MovieAnalytics
from src.transformer_model import load_transformer_model, predict_transformer_sentiment

app = FastAPI(
    title="Letterboxd Movie Sentiment API",
    description="API for sentiment analysis and movie-level analytics using Letterboxd-style reviews.",
    version="0.2.0",
)

# Load baseline model
vectorizer, baseline_model = load_model_and_vectorizer()

# Load analytics (still based on baseline for now)
analytics = MovieAnalytics()

# Try to load DistilBERT. If it is not trained yet, keep it disabled.
try:
    transformer_tokenizer, transformer_model = load_transformer_model()
    TRANSFORMER_AVAILABLE = True
    print("DistilBERT model loaded for transformer inference.")
except Exception as e:
    transformer_tokenizer, transformer_model = None, None
    TRANSFORMER_AVAILABLE = False
    print(f"DistilBERT model not available: {e}")


class TextRequest(BaseModel):
    text: str
    movie: Optional[str] = None
    model_type: str = "baseline"  # "baseline" or "transformer"


class MovieRequest(BaseModel):
    movie: str


class CompareMoviesRequest(BaseModel):
    movie_a: str
    movie_b: str


@app.get("/")
def root() -> dict:
    return {
        "message": "Letterboxd Movie Sentiment API. Visit /docs for interactive docs.",
        "models": {
            "baseline": True,
            "transformer": TRANSFORMER_AVAILABLE,
        },
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze-text")
def analyze_text(req: TextRequest) -> dict:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    model_type = req.model_type.lower()

    if model_type == "transformer":
        if not TRANSFORMER_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail="Transformer model not available. Train and save it first.",
            )
        result = predict_transformer_sentiment(
            text, transformer_tokenizer, transformer_model
        )
    else:
        # default to baseline
        result = predict_sentiment_for_text(text, vectorizer, baseline_model)

    response = {
        "text": text,
        "model_type": model_type,
        "sentiment": result["label"],
        "probability": result["probability"],
        "all_probs": result["all_probs"],
    }

    if req.movie:
        response["movie"] = req.movie

    return response


@app.post("/movie-summary")
def movie_summary(req: MovieRequest) -> dict:
    movie = req.movie.strip()
    if not movie:
        raise HTTPException(status_code=400, detail="Movie name must not be empty.")

    summary = analytics.get_movie_summary(movie)
    if summary.get("count", 0) == 0:
        raise HTTPException(
            status_code=404, detail=f"No reviews found for '{movie}'"
        )

    return summary


@app.post("/compare-movies")
def compare_movies(req: CompareMoviesRequest) -> dict:
    movie_a = req.movie_a.strip()
    movie_b = req.movie_b.strip()

    if not movie_a or not movie_b:
        raise HTTPException(
            status_code=400,
            detail="Both movie_a and movie_b must be provided.",
        )

    result = analytics.compare_movies(movie_a, movie_b)
    return result