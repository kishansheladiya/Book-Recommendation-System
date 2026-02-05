"""Configuration and constants for the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42

REQUIRED_RATINGS = ["user_id", "book_id", "rating"]
REQUIRED_BOOKS = ["book_id", "title", "authors", "average_rating", "publication_year"]
