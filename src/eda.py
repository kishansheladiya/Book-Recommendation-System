"""Exploratory Data Analysis: generate plots and save to reports/figures."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from .config import FIGURES_DIR
from .utils import get_logger

logger = get_logger("eda")


def plot_rating_distribution(ratings: pd.DataFrame, out: Path):
    plt.figure(figsize=(6,4))
    sns.histplot(ratings["rating"], bins=10)
    plt.title("Rating distribution")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out)
    plt.close()


def interactions_per_user(ratings: pd.DataFrame, out: Path):
    counts = ratings.groupby("user_id").size()
    plt.figure(figsize=(6,4))
    sns.histplot(counts, bins=50)
    plt.title("Interactions per user")
    plt.savefig(out)
    plt.close()


def interactions_per_book(ratings: pd.DataFrame, out: Path):
    counts = ratings.groupby("book_id").size()
    plt.figure(figsize=(6,4))
    sns.histplot(counts, bins=50)
    plt.title("Interactions per book")
    plt.savefig(out)
    plt.close()


def sparsity_analysis(ratings: pd.DataFrame, out: Path):
    users = ratings["user_id"].nunique()
    books = ratings["book_id"].nunique()
    sparsity = 1.0 - len(ratings) / (users * books)
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"users={users}\nbooks={books}\nsparsity={sparsity:.6f}\n")


def run(ratings_path: str):
    ratings = pd.read_csv(ratings_path)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_rating_distribution(ratings, FIGURES_DIR / "rating_dist.png")
    interactions_per_user(ratings, FIGURES_DIR / "interactions_per_user.png")
    interactions_per_book(ratings, FIGURES_DIR / "interactions_per_book.png")
    sparsity_analysis(ratings, FIGURES_DIR / "sparsity.txt")
    logger.info("EDA figures saved to %s", FIGURES_DIR)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: eda.py ratings.csv")
    else:
        run(sys.argv[1])
