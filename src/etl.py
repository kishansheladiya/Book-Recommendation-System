"""ETL: Clean ratings and books, merge metadata, normalize ids, save processed data."""
import pandas as pd
from pathlib import Path
from .config import RAW_DIR, PROCESSED_DIR
from .utils import get_logger, ensure_dirs

logger = get_logger("etl")


def clean_ratings(ratings_path: Path):
    df = pd.read_csv(ratings_path)
    logger.info("Loaded ratings: %d rows", len(df))
    # Keep only necessary columns
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.drop_duplicates()
    df = df.dropna(subset=["user_id", "book_id", "rating"]) 
    df["user_id"] = df["user_id"].astype(str)
    df["book_id"] = df["book_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    return df


def clean_books(books_path: Path):
    df = pd.read_csv(books_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df = df.drop_duplicates(subset=["book_id"]) 
    df["book_id"] = df["book_id"].astype(str)
    # Fill missing but keep description optional
    if "title" in df.columns:
        df["title"] = df["title"].fillna("")
    if "authors" in df.columns:
        df["authors"] = df["authors"].fillna("")
    if "publication_year" in df.columns:
        df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").fillna(0).astype(int)
    return df


def normalize_ids(ratings_df: pd.DataFrame, books_df: pd.DataFrame):
    # Map original ids to contiguous ints
    user_map = {u: i for i, u in enumerate(ratings_df["user_id"].unique())}
    book_map = {b: i for i, b in enumerate(books_df["book_id"].unique())}
    ratings_df["user_idx"] = ratings_df["user_id"].map(user_map)
    ratings_df["book_idx"] = ratings_df["book_id"].map(book_map)
    books_df["book_idx"] = books_df["book_id"].map(book_map)
    return ratings_df, books_df, user_map, book_map


def run(ratings_file: str, books_file: str):
    ensure_dirs()
    ratings_path = Path(ratings_file)
    books_path = Path(books_file)
    ratings = clean_ratings(ratings_path)
    books = clean_books(books_path)
    merged = ratings.merge(books, on="book_id", how="left", suffixes=("", "_book"))
    ratings, books, user_map, book_map = normalize_ids(merged[["user_id", "book_id", "rating"]], books)

    PROC_RATINGS = Path(PROCESSED_DIR) / "ratings_clean.csv"
    PROC_BOOKS = Path(PROCESSED_DIR) / "books_clean.csv"
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    ratings.to_csv(PROC_RATINGS, index=False)
    books.to_csv(PROC_BOOKS, index=False)

    logger.info("Saved processed ratings to %s", PROC_RATINGS)
    logger.info("Saved processed books to %s", PROC_BOOKS)
    return PROC_RATINGS, PROC_BOOKS


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: etl.py ratings.csv books.csv")
    else:
        run(sys.argv[1], sys.argv[2])
