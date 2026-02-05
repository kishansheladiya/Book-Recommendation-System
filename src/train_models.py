"""Train models: regression, classifier, kmeans, collaborative filtering, SVD."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import joblib

from .config import PROCESSED_DIR, MODELS_DIR, RANDOM_STATE
from .utils import get_logger, set_seed

logger = get_logger("train_models")


def prepare_train_test(train_path: str, test_path: str = None, seed: int = RANDOM_STATE):
    train = pd.read_csv(train_path)
    if test_path and Path(test_path).exists():
        test = pd.read_csv(test_path)
    else:
        # split by interactions (user-wise random split)
        rng = np.random.RandomState(seed)
        mask = rng.rand(len(train)) < 0.8
        test = train[~mask]
        train = train[mask]
    return train, test


def train_regression(train: pd.DataFrame, books: pd.DataFrame):
    # Simple features: book mean and user mean aggregated from train
    user_mean = train.groupby("user_id")["rating"].mean()
    book_mean = train.groupby("book_id")["rating"].mean()
    X = pd.DataFrame({
        "user_mean": train["user_id"].map(user_mean),
        "book_mean": train["book_id"].map(book_mean),
    }).fillna(0)
    y = train["rating"]
    model = LinearRegression()
    model.fit(X, y)
    return model


def train_classifier(train: pd.DataFrame):
    # label: liked if rating >=4
    df = train.copy()
    df["liked"] = (df["rating"] >= 4).astype(int)
    user_count = df.groupby("user_id")["rating"].count()
    user_mean = df.groupby("user_id")["rating"].mean()
    X = pd.DataFrame({
        "u_count": df["user_id"].map(user_count).fillna(0),
        "u_mean": df["user_id"].map(user_mean).fillna(df["rating"].mean()),
    })
    y = df["liked"]
    clf = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=6)
    clf.fit(X, y)
    return clf


def train_kmeans(books: pd.DataFrame, n_clusters=15):
    """K-Means clustering on book features; reduced clusters for speed."""
    X = books[[c for c in ["average_rating"] if c in books.columns]].fillna(0)
    if X.shape[1] == 0:
        X = pd.DataFrame({"dummy": np.zeros(len(books))})
    n_clust = min(n_clusters, len(books))
    kmeans = KMeans(n_clusters=n_clust, random_state=RANDOM_STATE, n_init=3, max_iter=100)
    kmeans.fit(X)
    return kmeans


def train_cf(train: pd.DataFrame):
    # Build item-user sparse matrix (items x users)
    pivot = train.pivot_table(index="user_id", columns="book_id", values="rating", fill_value=0)
    users = list(pivot.index)
    items = list(pivot.columns)
    # create sparse matrix
    mat = sparse.csr_matrix(pivot.values)
    # compute item similarity using sparse operations; transpose to items x users
    item_mat = mat.T
    # cosine_similarity supports sparse input and can return sparse if dense_output=False
    item_sim = cosine_similarity(item_mat, dense_output=False)
    return {"item_sim": item_sim, "items": items, "users": users}


def train_svd(train: pd.DataFrame, n_components=30):
    """Fit SVD with reduced components for speed."""
    pivot = train.pivot_table(index="user_id", columns="book_id", values="rating", fill_value=0)
    n_comp = min(n_components, min(pivot.shape)-1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE, n_iter=7)
    svd.fit(pivot.values)
    return svd, list(pivot.index), list(pivot.columns)


def run(train_path: str, test_path: str = None, books_path: str = None):
    set_seed(RANDOM_STATE)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)
    train, test = prepare_train_test(train_path, test_path)
    books = pd.read_csv(books_path) if books_path else pd.DataFrame()

    logger.info("Training regression model")
    reg = train_regression(train, books)
    joblib.dump(reg, Path(MODELS_DIR) / "regression.joblib")

    logger.info("Training classifier model")
    clf = train_classifier(train)
    joblib.dump(clf, Path(MODELS_DIR) / "classifier.joblib")

    logger.info("Training kmeans")
    kmeans = train_kmeans(books)
    joblib.dump(kmeans, Path(MODELS_DIR) / "kmeans.joblib")

    logger.info("Training collaborative filtering similarities")
    cf = train_cf(train)
    joblib.dump(cf, Path(MODELS_DIR) / "cf_joblib.joblib")

    logger.info("Training SVD")
    svd, users, items = train_svd(train)
    joblib.dump({"svd": svd, "users": users, "items": items}, Path(MODELS_DIR) / "svd.joblib")

    logger.info("All models trained and saved to %s", MODELS_DIR)
    return Path(MODELS_DIR)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: train_models.py train.csv books.csv [test.csv]")
    else:
        if len(sys.argv) >= 4:
            run(sys.argv[1], sys.argv[3], sys.argv[2])
        else:
            run(sys.argv[1], None, sys.argv[2])
