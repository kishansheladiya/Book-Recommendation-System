"""Utility helpers: logging, file helpers, seed setting."""
import logging
from pathlib import Path
import json
import random
import numpy as np
import os

from .config import RANDOM_STATE, RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR


def set_seed(seed: int = None):
    if seed is None:
        seed = RANDOM_STATE
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs():
    for p in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        Path(p).mkdir(parents=True, exist_ok=True)


def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
