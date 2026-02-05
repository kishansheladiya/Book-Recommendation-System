# Book Recommendation System (Goodreads dataset)

This repository implements an end-to-end machine learning pipeline for a book recommendation system using a local Goodreads archive (`archive.zip`). It focuses on ML modelling and evaluation rather than web deployment.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the pipeline end-to-end (extract data, ETL, train, evaluate):

```powershell
python run.py --do_eda True
```

Files of interest

- `src/download_data.py`: Extracts `archive.zip` to `data/raw/` and validates schema.
- `src/etl.py`: Cleans and normalizes data, saves to `data/processed/`.
- `src/features.py`: Builds user/book features and TF-IDF on titles.
- `src/train_models.py`: Trains regression, classifier, kmeans, CF and SVD.
- `src/evaluate.py`: Computes metrics and saves them to `reports/`.
- `src/recommender.py`: `recommend_for_user` and `similar_books` helper functions.
- `run.py`: Top-level script to run the pipeline.

Output

- Processed data: `data/processed/`
- Models: `models/`
- Reports and figures: `reports/`
