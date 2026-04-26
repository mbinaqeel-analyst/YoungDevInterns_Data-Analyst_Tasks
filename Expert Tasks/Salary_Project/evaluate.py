"""
src/evaluate.py
───────────────
Handles model evaluation:
  - Cross-validation (KFold)
  - Hold-out test set metrics (MAE, RMSE, R²)
  - Comparison DataFrame
  - Saves metrics to CSV
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline    import Pipeline
from sklearn.model_selection import (
    KFold, cross_validate
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils import get_logger, METRICS_PATH, COMPARISON_PATH

log = get_logger("evaluate")

# ── Cross-validation config ───────────────────────────────────────────────────
CV_FOLDS   = 5        # 5-fold CV on 250k rows = each fold ≈ 50k rows
CV_SCORING = {
    "mae":  "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2":   "r2",
}


def cross_validate_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> dict:
    """
    Run 5-fold cross-validation on the training set.

    Returns
    -------
    dict with cv_mae_mean, cv_mae_std, cv_r2_mean, cv_r2_std
    """
    log.info(f"  Running {CV_FOLDS}-fold CV for: {model_name}")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=kf,
        scoring=CV_SCORING,
        return_train_score=False,
        n_jobs=-1,
    )

    cv_mae  = -cv_results["test_mae"]
    cv_rmse = -cv_results["test_rmse"]
    cv_r2   =  cv_results["test_r2"]

    result = {
        "cv_mae_mean":  cv_mae.mean(),
        "cv_mae_std":   cv_mae.std(),
        "cv_rmse_mean": cv_rmse.mean(),
        "cv_rmse_std":  cv_rmse.std(),
        "cv_r2_mean":   cv_r2.mean(),
        "cv_r2_std":    cv_r2.std(),
    }

    log.info(
        f"  {model_name} CV → "
        f"MAE: ${result['cv_mae_mean']:,.0f} ± ${result['cv_mae_std']:,.0f}  |  "
        f"R²: {result['cv_r2_mean']:.4f} ± {result['cv_r2_std']:.4f}"
    )

    return result


def evaluate_on_test(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """
    Evaluate a fitted pipeline on the hold-out test set.

    Returns
    -------
    dict with test_mae, test_rmse, test_r2, y_pred array
    """
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    log.info(
        f"  {model_name} TEST → "
        f"MAE: ${mae:,.0f}  |  RMSE: ${rmse:,.0f}  |  R²: {r2:.4f}"
    )

    return {
        "model_name": model_name,
        "test_mae":   mae,
        "test_rmse":  rmse,
        "test_r2":    r2,
        "y_pred":     y_pred,
    }


def build_comparison_table(all_results: list[dict]) -> pd.DataFrame:
    """
    Combine CV + test results into a clean comparison DataFrame.

    Parameters
    ----------
    all_results : list of dicts, one per model,
                  each containing test + CV metric keys.
    """
    rows = []
    for r in all_results:
        rows.append({
            "Model":         r["model_name"],
            "Test MAE":      round(r["test_mae"],  2),
            "Test RMSE":     round(r["test_rmse"], 2),
            "Test R²":       round(r["test_r2"],   4),
            "CV MAE (mean)": round(r.get("cv_mae_mean", 0),  2),
            "CV MAE (std)":  round(r.get("cv_mae_std",  0),  2),
            "CV R² (mean)":  round(r.get("cv_r2_mean",  0),  4),
            "CV R² (std)":   round(r.get("cv_r2_std",   0),  4),
        })

    df = pd.DataFrame(rows).sort_values("Test R²", ascending=False)
    return df


def save_comparison(comparison_df: pd.DataFrame) -> None:
    """Save model comparison table and best-model metrics to disk."""
    comparison_df.to_csv(COMPARISON_PATH, index=False)
    log.info(f"Saved comparison → {COMPARISON_PATH}")

    # Best model row → separate metrics file (for Streamlit dashboard)
    best_row = comparison_df.iloc[0].to_dict()
    pd.DataFrame([best_row]).to_csv(METRICS_PATH, index=False)
    log.info(f"Saved best metrics → {METRICS_PATH}")


def print_leaderboard(comparison_df: pd.DataFrame) -> None:
    """Pretty-print the model leaderboard."""
    print("\n" + "═" * 72)
    print("  MODEL LEADERBOARD  (sorted by Test R²)")
    print("═" * 72)
    print(comparison_df.to_string(index=False))
    print("═" * 72)
    winner = comparison_df.iloc[0]
    print(f"\n  ✅  Best Model : {winner['Model']}")
    print(f"      Test R²   : {winner['Test R²']:.4f}")
    print(f"      Test MAE  : ${winner['Test MAE']:,.0f}")
    print(f"      Test RMSE : ${winner['Test RMSE']:,.0f}")
    print("═" * 72 + "\n")
