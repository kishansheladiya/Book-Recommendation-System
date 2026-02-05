"""Main runner script to execute the pipeline end-to-end."""
import argparse
import sys
from pathlib import Path

from src import download_data, etl, eda, train_models, evaluate
from src.utils import get_logger, ensure_dirs, set_seed
from src.config import RAW_DIR, PROCESSED_DIR, MODELS_DIR

logger = get_logger("run")


def main(args):
    set_seed()
    ensure_dirs()

    # Extract / ensure raw files
    ratings_path, books_path = download_data.extract_archive("archive.zip")

    # If user provided arguments override
    if args.train:
        ratings_path = Path(args.train)
    if args.books:
        books_path = Path(args.books)

    # ETL
    proc_ratings, proc_books = etl.run(str(ratings_path), str(books_path))

    # EDA
    if args.do_eda:
        eda.run(str(proc_ratings))

    # Train models
    train_models.run(str(proc_ratings), args.test if args.test else None, str(proc_books))

    # Evaluate (sample users if full dataset for speed)
    test_path = args.test if args.test else str(proc_ratings)
    sample_eval = 1000 if not args.test else None  # Sample 1000 users on full dataset eval
    evaluate.run_all(str(proc_ratings), test_path, str(proc_books), k=args.k, sample_eval_users=sample_eval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Book Recommender pipeline")
    parser.add_argument("--train", help="train ratings csv", default=None)
    parser.add_argument("--test", help="test ratings csv", default=None)
    parser.add_argument("--books", help="books csv", default=None)
    parser.add_argument("--mode", help="mode classification|regression", default="classification")
    parser.add_argument("--do_eda", help="run EDA", action="store_true")
    parser.add_argument("--k", help="k for recommendation metrics", type=int, default=10)
    args = parser.parse_args()
    main(args)
