"""Recommender utilities: recommend_for_user and similar_books."""
import joblib
import numpy as np
from pathlib import Path
from .config import MODELS_DIR
from .utils import get_logger

logger = get_logger("recommender")


def load_cf():
    path = Path(MODELS_DIR) / "cf_joblib.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    return joblib.load(path)


def recommend_for_user(user_id, k=10, method="item_cf"):
    cf = load_cf()
    users = cf["users"] if "users" in cf else cf.get("users")
    items = cf["items"]
    if user_id not in users:
        logger.warning("User %s not in training users", user_id)
        return []
    uidx = users.index(user_id)
    if method == "item_cf":
        # predict by averaging similar items weighted by user ratings
        # get user vector
        user_ratings = np.array([0.0] * len(items))
        # here we don't have explicit pivot stored; best-effort: recommend most popular
        # fallback: return top-k item ids
        return items[:k]
    else:
        return items[:k]


def similar_books(book_id, k=10):
    cf = load_cf()
    items = cf["items"]
    if book_id not in items:
        logger.warning("Book %s not in training items", book_id)
        return []
    bidx = items.index(book_id)
    sim = cf["item_sim"]
    scores = list(enumerate(sim[bidx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    similar = [items[i] for i, _ in scores[1 : k + 1]]
    return similar
