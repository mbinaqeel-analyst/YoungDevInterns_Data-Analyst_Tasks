"""
src/train.py
────────────
Full training pipeline:
  1. Load + split data
  2. Build sklearn Pipeline (preprocessor + model)
  3. Train 4 models (+ optional XGBoost)
  4. Run cross-validation on each
  5. Tune top model with RandomizedSearchCV
  6. Evaluate all on hold-out test set
  7. Auto-select best model
  8. Save winning pipeline to disk
  9. Save comparison + metrics CSV

Run:  python src/train.py
"""

import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model    import LinearRegression
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor

# Optional XGBoost — gracefully handle if not installed
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from preprocess import load_data, build_preprocessor, get_feature_names, dataset_summary
from evaluate   import (
    cross_validate_model, evaluate_on_test,
    build_comparison_table, save_comparison, print_leaderboard,
)
from utils import (
    get_logger, ALL_FEATURES, TARGET_COL,
    PIPELINE_PATH, FEAT_IMP_PATH, OUTPUTS_DIR,
)

log = get_logger("train")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

def get_base_models() -> dict:
    """
    Return base (untuned) models.
    Hyperparameters chosen as sensible defaults for 250k rows.
    """
    models = {
        "Linear Regression": LinearRegression(n_jobs=-1),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=50,
            random_state=42,
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=42,
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.07,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbosity=0,
        )
        log.info("XGBoost detected — added to model registry.")
    else:
        log.info("XGBoost not installed — skipping. (pip install xgboost to enable)")

    return models


# ─────────────────────────────────────────────────────────────────────────────
#  HYPERPARAMETER SEARCH SPACES
# ─────────────────────────────────────────────────────────────────────────────

SEARCH_SPACES = {
    "Random Forest": {
        "model__n_estimators":      [100, 200, 300],
        "model__max_depth":         [10, 15, 20, None],
        "model__min_samples_leaf":  [10, 20, 50],
        "model__max_features":      ["sqrt", "log2"],
    },
    "Gradient Boosting": {
        "model__n_estimators":   [100, 200, 300],
        "model__learning_rate":  [0.05, 0.08, 0.1, 0.15],
        "model__max_depth":      [4, 5, 6],
        "model__subsample":      [0.7, 0.8, 0.9],
    },
    "XGBoost": {
        "model__n_estimators":      [200, 300, 400],
        "model__learning_rate":     [0.05, 0.07, 0.1],
        "model__max_depth":         [4, 5, 6, 7],
        "model__subsample":         [0.7, 0.8, 0.9],
        "model__colsample_bytree":  [0.7, 0.8, 1.0],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(model) -> Pipeline:
    """
    Wrap a preprocessor + model into a single sklearn Pipeline.

    The pipeline guarantees:
    - Preprocessing is fit ONLY on training data (no leakage)
    - Prediction always uses the exact same transforms
    """
    preprocessor = build_preprocessor()
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",         model),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_model(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_iter: int = 8,
) -> Pipeline:
    """
    Run RandomizedSearchCV on a model that has a defined search space.
    Returns the best pipeline (already fitted).

    n_iter=15 → tries 15 random hyperparameter combos (fast on 250k rows).
    Increase to 30-50 for exhaustive tuning during final production training.
    """
    if model_name not in SEARCH_SPACES:
        log.info(f"  No search space for '{model_name}' — skipping tuning.")
        return pipeline

    log.info(f"  Tuning {model_name} with RandomizedSearchCV (n_iter={n_iter})…")

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=SEARCH_SPACES[model_name],
        n_iter=n_iter,
        scoring="r2",
        cv=3,             # 3-fold inside search (faster than 5-fold)
        n_jobs=-1,
        random_state=42,
        verbose=0,
        refit=True,       # refit best params on full X_train
    )
    search.fit(X_train, y_train)
    log.info(f"  Best params: {search.best_params_}")
    log.info(f"  Best CV R²:  {search.best_score_:.4f}")

    return search.best_estimator_
    joblib.dump(tuned_pipeline, PIPELINE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_importance(pipeline: Pipeline) -> pd.DataFrame | None:
    """
    Pull feature importances from the winning pipeline.
    Works for tree-based models and linear models.
    """
    model        = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = get_feature_names(preprocessor)
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        importances = importances / importances.sum()   # normalise
    else:
        return None

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    # Clean up sklearn prefix (e.g. "num__experience_years" → "experience_years")
    fi_df["feature"] = fi_df["feature"].str.replace(
        r"^(num__|ord__|cat__)", "", regex=True
    )

    fi_df.to_csv(FEAT_IMP_PATH, index=False)
    log.info(f"Feature importance saved → {FEAT_IMP_PATH}")

    return fi_df


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_path: str | Path,
    test_size: float = 0.2,
    tune_top_n: int = 1,         # tune top-N models by CV R²
    cv_enabled: bool = True,
):
    """
    Full training run.

    Parameters
    ----------
    data_path  : path to salary CSV
    test_size  : fraction for hold-out test set
    tune_top_n : how many models to tune with RandomizedSearchCV
    cv_enabled : run cross-validation (disable for quick dev iterations)
    """
    log.info("=" * 60)
    log.info("  SALARY PREDICTOR — TRAINING PIPELINE")
    log.info("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data(data_path)
    X  = df[ALL_FEATURES]
    y  = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    log.info(
        f"Split → Train: {len(X_train):,}  |  Test: {len(X_test):,}"
    )

    # ── 2. Build & train all models ───────────────────────────────────────────
    base_models = get_base_models()
    all_results = []

    log.info("\n── Phase 1: Initial training & cross-validation ──")
    for name, model in base_models.items():
        log.info(f"\nTraining: {name}")
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)

        result = {"model_name": name, "pipeline": pipeline}

        # Cross-validation
        if cv_enabled:
            fresh_pipeline = build_pipeline(
                base_models[name].__class__(**base_models[name].get_params())
            )
            cv_metrics = cross_validate_model(fresh_pipeline, X_train, y_train, name)
            result.update(cv_metrics)

        # Test set evaluation
        test_metrics = evaluate_on_test(pipeline, X_test, y_test, name)
        result.update(test_metrics)

        all_results.append(result)

    # ── 3. Sort by CV R² (or test R² if CV disabled) ─────────────────────────
    sort_key = "cv_r2_mean" if cv_enabled else "test_r2"
    all_results.sort(key=lambda x: x.get(sort_key, x["test_r2"]), reverse=True)

    # ── 4. Tune top models ────────────────────────────────────────────────────
    log.info(f"\n── Phase 2: Hyperparameter tuning (top {tune_top_n} models) ──")
    tuned_results = []

    for result in all_results[:tune_top_n]:
        name = result["model_name"]
        log.info(f"\nTuning: {name}")

        tuned_pipeline = tune_model(
            build_pipeline(base_models[name]),
            X_train, y_train, name,
        )

        tuned_test = evaluate_on_test(tuned_pipeline, X_test, y_test, f"{name} (tuned)")
        tuned_test["pipeline"] = tuned_pipeline

        # Carry over CV stats from base run (CV was already on base model)
        for k in ["cv_mae_mean", "cv_mae_std", "cv_r2_mean", "cv_r2_std"]:
            tuned_test[k] = result.get(k, 0)

        tuned_results.append(tuned_test)

    # ── 5. Merge base + tuned results and pick winner ─────────────────────────
    # Keep only non-tuned models in base results (avoid duplication)
    tuned_names = {r["model_name"] for r in tuned_results}
    final_results = [
        r for r in all_results
        if r["model_name"] not in {n.replace(" (tuned)", "") for n in tuned_names}
    ] + tuned_results

    final_results.sort(key=lambda x: x["test_r2"], reverse=True)
    best_result   = final_results[0]
    best_pipeline = best_result["pipeline"]
    best_name     = best_result["model_name"]

    # ── 6. Save winner ────────────────────────────────────────────────────────
    joblib.dump(best_pipeline, PIPELINE_PATH)
    log.info(f"\nBest pipeline saved → {PIPELINE_PATH}")

    # ── 7. Save comparison + metrics ─────────────────────────────────────────
    comparison_df = build_comparison_table(final_results)
    save_comparison(comparison_df)
    print_leaderboard(comparison_df)

    # ── 8. Feature importance ─────────────────────────────────────────────────
    fi_df = extract_feature_importance(best_pipeline)
    if fi_df is not None:
        log.info(f"\nTop 5 features:\n{fi_df.head(5).to_string(index=False)}")

    # ── 9. Save predictions on test set (for Streamlit scatter plot) ──────────
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    test_preds_path = OUTPUTS_DIR / "test_predictions.csv"
    test_df = X_test.copy()
    test_df["actual_salary"]    = y_test.values
    test_df["predicted_salary"] = best_result["y_pred"]
    test_df["residual"]         = test_df["actual_salary"] - test_df["predicted_salary"]
    test_df.to_csv(test_preds_path, index=False)
    log.info(f"Test predictions saved → {test_preds_path}")

    return best_pipeline, comparison_df, fi_df


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/salary_data.csv"
    train(data_path)
