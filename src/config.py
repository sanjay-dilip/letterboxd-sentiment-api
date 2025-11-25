from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
RAW_LETTERBOXD_PATH = RAW_DATA_DIR / "letterboxd-reviews.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "cleaned_reviews.csv"