"""Download / extract local archive and validate required files."""
import zipfile
from pathlib import Path
import shutil
import pandas as pd
from .config import RAW_DIR, REQUIRED_RATINGS, REQUIRED_BOOKS
from .utils import get_logger, ensure_dirs

logger = get_logger("download_data")


def extract_archive(archive_path: str = "archive.zip"):
    ensure_dirs()
    archive = Path(archive_path)
    if not archive.exists():
        logger.error("Archive not found: %s", archive)
        raise FileNotFoundError(archive)

    # If required files already present, do nothing
    ratings_path = RAW_DIR / "ratings.csv"
    books_path = RAW_DIR / "books.csv"
    if ratings_path.exists() and books_path.exists():
        logger.info("Required files already exist in %s", RAW_DIR)
        return ratings_path, books_path

    logger.info("Extracting %s to %s", archive, RAW_DIR)
    with zipfile.ZipFile(archive, "r") as z:
        z.extractall(RAW_DIR)

    # Search for files in extracted tree
    found_ratings = None
    found_books = None
    for p in RAW_DIR.rglob("*.csv"):
        name = p.name.lower()
        if name == "ratings.csv":
            found_ratings = p
        if name == "books.csv":
            found_books = p

    if found_ratings:
        shutil.move(str(found_ratings), str(ratings_path))
    if found_books:
        shutil.move(str(found_books), str(books_path))

    # Clean up any nested extracted folders if empty
    for child in RAW_DIR.iterdir():
        if child.is_dir():
            try:
                next(child.iterdir())
            except StopIteration:
                child.rmdir()

    # Validate
    validate_schema(ratings_path, books_path)
    return ratings_path, books_path


def validate_schema(ratings_file: Path, books_file: Path):
    logger.info("Validating schemas")
    if not ratings_file.exists() or not books_file.exists():
        raise FileNotFoundError("ratings.csv or books.csv not found after extraction")

    r = pd.read_csv(ratings_file, nrows=5)
    b = pd.read_csv(books_file, nrows=5)

    for c in REQUIRED_RATINGS:
        if c not in r.columns:
            raise ValueError(f"ratings.csv missing required column: {c}")
    for c in REQUIRED_BOOKS:
        if c not in b.columns:
            logger.warning("books.csv missing optional required column: %s", c)

    logger.info("Schema validation passed")


if __name__ == "__main__":
    extract_archive()
