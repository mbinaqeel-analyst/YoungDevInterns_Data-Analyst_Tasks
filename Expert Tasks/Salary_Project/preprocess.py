"""
src/preprocess.py
─────────────────
Builds a reusable sklearn Pipeline that handles:
  - Numerical features  → median imputation + StandardScaler
  - Ordinal features    → ordinal imputation + OrdinalEncoder (order-aware)
  - Binary features     → most-frequent imputation + OneHotEncoder
  - Nominal features    → most-frequent imputation + OneHotEncoder(handle_unknown='ignore')

KEY DESIGN DECISION:
  The entire preprocessing logic lives inside one sklearn Pipeline object.
  This means:
  ✅ No data leakage — fit() only sees training data
  ✅ Prediction reuses the exact same transforms automatically
  ✅ One joblib file holds everything (pipeline = preprocessor + model)
  ✅ Unknown categories (new cities, job titles) are handled safely
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline           import Pipeline
from sklearn.compose            import ColumnTransformer
from sklearn.preprocessing      import (
    OrdinalEncoder, OneHotEncoder, StandardScaler
)
from sklearn.impute             import SimpleImputer

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    get_logger,
    NUMERICAL_FEATURES, ORDINAL_FEATURES,
    BINARY_FEATURES, NOMINAL_FEATURES,
    TARGET_COL, ALL_FEATURES,
)

log = get_logger("preprocess")


# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load CSV and run basic validation."""
    log.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    log.info(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ── Check required columns exist ─────────────────────────────────────────
    expected = set(ALL_FEATURES + [TARGET_COL])
    missing  = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    # ── Log target stats ──────────────────────────────────────────────────────
    log.info(
        f"Salary → mean: ${df[TARGET_COL].mean():,.0f} "
        f"| median: ${df[TARGET_COL].median():,.0f} "
        f"| min: ${df[TARGET_COL].min():,.0f} "
        f"| max: ${df[TARGET_COL].max():,.0f}"
    )

    # ── Log missing value summary ────────────────────────────────────────────
    n_missing = df[ALL_FEATURES + [TARGET_COL]].isnull().sum()
    n_missing = n_missing[n_missing > 0]
    if len(n_missing):
        log.warning(f"Missing values detected:\n{n_missing.to_string()}")
    else:
        log.info("No missing values detected.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles all 4 feature types correctly.

    Returns
    -------
    ColumnTransformer (unfitted)
        Fit this on X_train ONLY.  apply .transform() on X_test / new input.
    """

    # ── 1. Numerical pipeline ────────────────────────────────────────────────
    #    impute → scale  (StandardScaler helps Linear Regression converge faster)
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    # ── 2. Ordinal pipeline ───────────────────────────────────────────────────
    #    impute with most-frequent → ordinal encode in defined category order
    ordinal_categories = [cats for cats in ORDINAL_FEATURES.values()]
    ord_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1,   # unseen ordinal values → -1 (safe fallback)
        )),
    ])

    # ── 3. Binary + Nominal pipeline ─────────────────────────────────────────
    #    impute → one-hot  (drop='first' to avoid multicollinearity)
    #    handle_unknown='ignore' → unseen category = all-zero row (safe)
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            drop="first",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("ord", ord_pipeline, list(ORDINAL_FEATURES.keys())),
            ("cat", cat_pipeline, BINARY_FEATURES + NOMINAL_FEATURES),
        ],
        remainder="drop",       # drop any extra columns not in schema
        verbose_feature_names_out=True,
    )

    return preprocessor


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE NAMES (post-transform)
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Extract human-readable feature names after fitting the ColumnTransformer.
    Needed for feature importance mapping.
    """
    return list(preprocessor.get_feature_names_out())


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK DATASET SUMMARY (used in Streamlit dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    """Return key statistics about the raw dataset."""
    return {
        "n_rows":          len(df),
        "n_features":      len(ALL_FEATURES),
        "salary_mean":     df[TARGET_COL].mean(),
        "salary_median":   df[TARGET_COL].median(),
        "salary_min":      df[TARGET_COL].min(),
        "salary_max":      df[TARGET_COL].max(),
        "salary_std":      df[TARGET_COL].std(),
        "n_job_titles":    df["job_title"].nunique(),
        "n_industries":    df["industry"].nunique(),
        "n_locations":     df["location"].nunique(),
        "pct_remote":      (df["remote_work"].str.lower() == "yes").mean(),
        "pct_certified":   (df["certifications"].str.lower() == "yes").mean(),
    }
