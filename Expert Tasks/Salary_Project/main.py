"""
main.py
───────
Master entry point — runs the full training pipeline.

Usage:
    python main.py                         # uses default data path
    python main.py data/salary_data.csv    # explicit path
    python main.py data/salary_data.csv --no-cv  # skip CV for speed
"""

import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "src")

from pathlib import Path
from train import train
from utils import get_logger

log = get_logger("main")


if __name__ == "__main__":
    args      = sys.argv[1:]
    data_path = args[0] if args and not args[0].startswith("--") else "data/salary_data.csv"
    cv_flag   = "--no-cv" not in args

    if not Path(data_path).exists():
        log.error(f"Data file not found: {data_path}")
        log.error("Place your Kaggle CSV at data/salary_data.csv and re-run.")
        sys.exit(1)

    log.info(f"Starting training pipeline → data: {data_path}")
    log.info(f"Cross-validation: {'enabled' if cv_flag else 'disabled'}")

    best_pipeline, comparison, fi_df = train(
        data_path=data_path,
        test_size=0.2,
        tune_top_n=1,
        cv_enabled=cv_flag,
    )

    print("\n" + "█" * 60)
    print("  PIPELINE COMPLETE")
    print("█" * 60)
    print("\n  Artifacts saved:")
    print("    models/salary_pipeline.pkl      ← trained model")
    print("    models/model_comparison.csv     ← all model results")
    print("    models/model_metrics.csv        ← best model metrics")
    print("    models/feature_importance.csv   ← feature importances")
    print("    outputs/test_predictions.csv    ← test set predictions")
    print("\n  Next steps:")
    print("    streamlit run app.py            ← launch web app")
    print("    python src/predict.py           ← test predictions")
    print("█" * 60 + "\n")
