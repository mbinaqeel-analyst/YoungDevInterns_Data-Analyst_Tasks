"""
src/utils.py
────────────
Shared constants, path management, and logging setup.
Import this at the top of every other module.
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime

# ── Project root (one level above src/) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Standard directories ──────────────────────────────────────────────────────
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ── Artifact file paths ───────────────────────────────────────────────────────
PIPELINE_PATH    = MODELS_DIR / "salary_pipeline.pkl"
METRICS_PATH     = MODELS_DIR / "model_metrics.csv"
FEAT_IMP_PATH    = MODELS_DIR / "feature_importance.csv"
COMPARISON_PATH  = MODELS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"

# ── Dataset column definitions ────────────────────────────────────────────────
TARGET_COL = "salary"

NUMERICAL_FEATURES = ["experience_years", "skills_count"]

ORDINAL_FEATURES = {
    "education_level": ["High School", "Associate", "Bachelor's", "Master's", "PhD"],
    "company_size":    ["Small", "Medium", "Large"],
}

BINARY_FEATURES = ["remote_work", "certifications"]   # Yes/No → treated as nominal

NOMINAL_FEATURES = ["job_title", "industry", "location"]

ALL_FEATURES = (
    NUMERICAL_FEATURES
    + list(ORDINAL_FEATURES.keys())
    + BINARY_FEATURES
    + NOMINAL_FEATURES
)


# ── Logging ───────────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ── Directory bootstrap ───────────────────────────────────────────────────────
def ensure_dirs() -> None:
    """Create all required project directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── Metric formatting helpers ─────────────────────────────────────────────────
def fmt_currency(value: float) -> str:
    return f"${value:,.0f}"


def fmt_pct(value: float, decimals: int = 2) -> str:
    return f"{value * 100:.{decimals}f}%"


# ── Safe JSON serialiser (handles numpy types) ────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


# ── Run once on import ────────────────────────────────────────────────────────
ensure_dirs()
