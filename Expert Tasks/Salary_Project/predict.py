"""
src/predict.py
──────────────
Prediction interface for the trained salary pipeline.

Provides:
  predict_one()   → single candidate dict → salary + band + range
  predict_batch() → full DataFrame → vectorized (no iterrows!)

Both functions load the pipeline once and apply it directly —
no manual encoding, no column alignment code.
sklearn Pipeline handles everything automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib

from utils import (
    get_logger, ALL_FEATURES, TARGET_COL,
    PIPELINE_PATH, PREDICTIONS_PATH, OUTPUTS_DIR,
    fmt_currency,
)

log = get_logger("predict")

# ── Salary band thresholds (in USD) ──────────────────────────────────────────
SALARY_BANDS = [
    (0,       50_000,  "Entry Level",        "🔴"),
    (50_000,  85_000,  "Mid Level",          "🟡"),
    (85_000,  130_000, "Senior Level",       "🔵"),
    (130_000, float("inf"), "Executive / Principal", "🟢"),
]

# ── Confidence range multiplier ───────────────────────────────────────────────
# A rough ±12% range is a reasonable heuristic for salary estimates
CONFIDENCE_PCT = 0.12


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline():
    """Load the trained sklearn pipeline from disk."""
    if not PIPELINE_PATH.exists():
        raise FileNotFoundError(
            f"No trained pipeline found at {PIPELINE_PATH}.\n"
            "Run `python src/train.py` to train the model first."
        )
    return joblib.load(PIPELINE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  SALARY BAND HELPER
# ─────────────────────────────────────────────────────────────────────────────

def classify_band(salary: float) -> tuple[str, str]:
    """Return (band_label, band_emoji) for a given salary value."""
    for low, high, label, emoji in SALARY_BANDS:
        if low <= salary < high:
            return label, emoji
    return "Executive / Principal", "🟢"


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

def predict_one(raw_input: dict, pipeline=None) -> dict:
    """
    Predict salary for one candidate.

    Parameters
    ----------
    raw_input : dict with keys matching ALL_FEATURES
    pipeline  : pre-loaded pipeline (optional — avoids disk reload in loops)

    Returns
    -------
    dict:
        predicted_salary  float
        salary_low        float  (–12% confidence range)
        salary_high       float  (+12% confidence range)
        salary_band       str
        band_emoji        str
        formatted         str    e.g. "$92,500"
        range_formatted   str    e.g. "$81,400 – $103,600"
    """
    if pipeline is None:
        pipeline = load_pipeline()

    # Build single-row DataFrame — sklearn Pipeline expects a DataFrame
    X = pd.DataFrame([raw_input])[ALL_FEATURES]

    salary = float(pipeline.predict(X)[0])
    salary = max(salary, 0.0)   # salary cannot be negative

    low  = salary * (1 - CONFIDENCE_PCT)
    high = salary * (1 + CONFIDENCE_PCT)
    band, emoji = classify_band(salary)

    return {
        "predicted_salary": round(salary, 2),
        "salary_low":       round(low, 2),
        "salary_high":      round(high, 2),
        "salary_band":      band,
        "band_emoji":       emoji,
        "formatted":        fmt_currency(salary),
        "range_formatted":  f"{fmt_currency(low)} – {fmt_currency(high)}",
    }


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH PREDICTION (vectorized — fast on 250k rows)
# ─────────────────────────────────────────────────────────────────────────────

def predict_batch(
    df: pd.DataFrame,
    pipeline=None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Run vectorized predictions on an entire DataFrame.

    REPLACES the slow iterrows() loop from the previous version.
    sklearn's pipeline.predict() processes all rows at once using
    numpy matrix operations — typically 100–500× faster.

    Parameters
    ----------
    df       : DataFrame with feature columns (salary column optional)
    pipeline : pre-loaded pipeline
    save     : if True, save enriched CSV to outputs/predictions.csv

    Returns
    -------
    Enriched DataFrame with columns:
        predicted_salary, salary_low, salary_high, salary_band, salary_gap
    """
    if pipeline is None:
        pipeline = load_pipeline()

    log.info(f"Running batch prediction on {len(df):,} rows…")

    X = df[ALL_FEATURES]   # select only model features, drop anything else

    # ── Vectorized prediction (single numpy call) ─────────────────────────────
    predictions = pipeline.predict(X)
    predictions = np.maximum(predictions, 0)   # clip negatives

    out = df.copy()
    out["predicted_salary"] = np.round(predictions, 2)
    out["salary_low"]       = np.round(predictions * (1 - CONFIDENCE_PCT), 2)
    out["salary_high"]      = np.round(predictions * (1 + CONFIDENCE_PCT), 2)

    # Salary band (vectorized via numpy select)
    thresholds  = [50_000, 85_000, 130_000]
    band_labels = ["Entry Level", "Mid Level", "Senior Level", "Executive / Principal"]
    conditions  = [
        predictions < thresholds[0],
        (predictions >= thresholds[0]) & (predictions < thresholds[1]),
        (predictions >= thresholds[1]) & (predictions < thresholds[2]),
    ]
    out["salary_band"] = np.select(conditions, band_labels[:3], default=band_labels[3])

    # Salary gap (requires actual salary column)
    if TARGET_COL in out.columns:
        out["salary_gap"]     = out[TARGET_COL] - out["predicted_salary"]
        out["salary_gap_pct"] = (out["salary_gap"] / out[TARGET_COL] * 100).round(2)

    log.info(
        f"Batch done → avg predicted: ${predictions.mean():,.0f} "
        f"| min: ${predictions.min():,.0f} | max: ${predictions.max():,.0f}"
    )

    if save:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        out.to_csv(PREDICTIONS_PATH, index=False)
        log.info(f"Predictions saved → {PREDICTIONS_PATH}")

    return out


# ─────────────────────────────────────────────────────────────────────────────
#  QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = load_pipeline()

    examples = [
        dict(job_title="Data Scientist", experience_years=7, education_level="Master's",
             skills_count=10, industry="Technology", company_size="Large",
             location="New York", remote_work="Yes", certifications="Yes"),

        dict(job_title="HR Manager", experience_years=2, education_level="Bachelor's",
             skills_count=4, industry="Healthcare", company_size="Small",
             location="Austin", remote_work="No", certifications="No"),

        dict(job_title="DevOps Engineer", experience_years=5, education_level="Bachelor's",
             skills_count=8, industry="Finance", company_size="Medium",
             location="San Francisco", remote_work="Yes", certifications="Yes"),
    ]

    print("\n" + "=" * 60)
    print("  SALARY PREDICTION EXAMPLES")
    print("=" * 60)

    for i, ex in enumerate(examples, 1):
        r = predict_one(ex, pipeline)
        print(f"\n  [{i}] {ex['job_title']} · {ex['experience_years']} yrs · {ex['education_level']}")
        print(f"      Salary  : {r['formatted']}")
        print(f"      Range   : {r['range_formatted']}")
        print(f"      Band    : {r['band_emoji']} {r['salary_band']}")
