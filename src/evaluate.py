"""Evaluation routines for regression, classification and recommendation metrics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
from .recommender import recommend_for_user
from .config import REPORTS_DIR
from .utils import get_logger, save_json

logger = get_logger("evaluate")


def classification_metrics(y_true, y_pred_proba, y_pred_label):
    metrics = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_pred_proba))
    except Exception:
        metrics["roc_auc"] = None
    metrics["f1"] = float(f1_score(y_true, y_pred_label))
    metrics["precision"] = float(precision_score(y_true, y_pred_label))
    metrics["recall"] = float(recall_score(y_true, y_pred_label))
    return metrics


def regression_metrics(y_true, y_pred):
    return {"rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))), "mae": float(mean_absolute_error(y_true, y_pred))}


def precision_recall_at_k(test_df: pd.DataFrame, k=10, sample_users=None):
    """Compute precision and recall@k. Optionally sample users for speed."""
    # test_df should have user_id, book_id, rating
    users = test_df["user_id"].unique()
    
    # Sample users if specified (for faster eval on large datasets)
    if sample_users and len(users) > sample_users:
        np.random.seed(42)
        users = np.random.choice(users, size=sample_users, replace=False)
    
    precisions = []
    recalls = []
    for u in users:
        true_pos = set(test_df[(test_df["user_id"] == u) & (test_df["rating"] >= 4)]["book_id"].astype(str).tolist())
        if not true_pos:
            continue
        recs = recommend_for_user(u, k=k)
        recs = [str(x) for x in recs]
        tp = len(set(recs) & true_pos)
        precisions.append(tp / k)
        recalls.append(tp / len(true_pos))
    if not precisions:
        return {"precision_at_k": None, "recall_at_k": None}
    return {"precision_at_k": float(np.mean(precisions)), "recall_at_k": float(np.mean(recalls))}


def run_all(train_path: str, test_path: str, books_path: str, k=10, sample_eval_users=None):
    """Run all evaluations. sample_eval_users: if set, evaluate on subset for speed."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path) if Path(test_path).exists() else pd.DataFrame()
    reports = {}

    # Classification evaluation
    from joblib import load

    clf_path = Path("models") / "classifier.joblib"
    if clf_path.exists():
        clf = load(clf_path)
        # prepare features for test
        if not test.empty:
            user_count = train.groupby("user_id")["rating"].count()
            user_mean = train.groupby("user_id")["rating"].mean()
            X_test = pd.DataFrame({
                "u_count": test["user_id"].map(user_count).fillna(0),
                "u_mean": test["user_id"].map(user_mean).fillna(train["rating"].mean()),
            })
            y_true = (test["rating"] >= 4).astype(int)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred
            reports["classification"] = classification_metrics(y_true, y_proba, y_pred)

    # Regression evaluation
    reg_path = Path("models") / "regression.joblib"
    if reg_path.exists() and not test.empty:
        reg = load(reg_path)
        user_mean = train.groupby("user_id")["rating"].mean()
        book_mean = train.groupby("book_id")["rating"].mean()
        X_test = pd.DataFrame({
            "user_mean": test["user_id"].map(user_mean).fillna(train["rating"].mean()),
            "book_mean": test["book_id"].map(book_mean).fillna(train["rating"].mean()),
        })
        y_pred = reg.predict(X_test)
        reports["regression"] = regression_metrics(test["rating"], y_pred)

    # Recommendation metrics (with optional sampling)
    if not test.empty:
        recs = precision_recall_at_k(test, k=k, sample_users=sample_eval_users)
        reports["recommendation"] = recs

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(reports, REPORTS_DIR / "metrics.json")
    with open(REPORTS_DIR / "metrics.md", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    logger.info("Saved metrics to %s", REPORTS_DIR)
    return reports


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: evaluate.py train.csv test.csv books.csv")
    else:
        run_all(sys.argv[1], sys.argv[2], sys.argv[3])
