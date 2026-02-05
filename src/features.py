"""Feature engineering: user/book features and optional TF-IDF on titles."""
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import PROCESSED_DIR, RANDOM_STATE
from .utils import get_logger

logger = get_logger("features")


class FeatureBuilder:
    def __init__(self, n_title_features: int = 50):
        self.tfidf = TfidfVectorizer(max_features=n_title_features, stop_words="english")
        self.fitted = False

    def fit(self, ratings_df: pd.DataFrame, books_df: pd.DataFrame):
        # User features
        user_stats = ratings_df.groupby("user_id")["rating"].agg(["count", "mean"]).rename(columns={"count": "u_count", "mean": "u_mean"})
        # Book features
        book_stats = ratings_df.groupby("book_id")["rating"].agg(["count", "mean"]).rename(columns={"count": "b_count", "mean": "b_mean"})
        books_df = books_df.set_index("book_id").join(book_stats, how="left").fillna({"b_count": 0, "b_mean": books_df.get("average_rating", 0)})

        # TF-IDF on titles
        titles = books_df["title"].fillna("")
        self.tfidf.fit(titles)

        self.user_stats = user_stats
        self.books_df = books_df
        self.fitted = True
        logger.info("FeatureBuilder fitted: users=%d books=%d", len(user_stats), len(books_df))

    def transform(self, ratings_df: pd.DataFrame, books_df: pd.DataFrame):
        if not self.fitted:
            raise RuntimeError("FeatureBuilder not fitted")
        user_feat = ratings_df["user_id"].map(self.user_stats["u_count"]).fillna(0)
        user_mean = ratings_df["user_id"].map(self.user_stats["u_mean"]).fillna(ratings_df["rating"].mean())
        book_feat_count = ratings_df["book_id"].map(self.books_df["b_count"]).fillna(0)
        book_mean = ratings_df["book_id"].map(self.books_df["b_mean"]).fillna(ratings_df["rating"].mean())

        title_tfidf = self.tfidf.transform(books_df["title"].fillna("")).toarray()

        X = pd.DataFrame({
            "u_count": user_feat.values,
            "u_mean": user_mean.values,
            "b_count": book_feat_count.values,
            "b_mean": book_mean.values,
        })
        # Attach title features with numeric column names
        tf_cols = {i: f"title_tfidf_{i}" for i in range(title_tfidf.shape[1])}
        tf_df = pd.DataFrame(title_tfidf).rename(columns=tf_cols)
        X = pd.concat([X.reset_index(drop=True), tf_df.reset_index(drop=True)], axis=1)
        return X


def fit_and_save(train_ratings_path: str, books_path: str, out_path: str = None):
    ratings = pd.read_csv(train_ratings_path)
    books = pd.read_csv(books_path)
    fb = FeatureBuilder()
    fb.fit(ratings, books)
    if out_path:
        pd.to_pickle(fb, out_path)
    return fb
