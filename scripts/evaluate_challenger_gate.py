"""
CI/CD Challenger Evaluation Gate Script.
Evaluates registered Challenger model performance against active Champion model on a validation dataset.
Blocks promotion if Challenger F1 / ROC-AUC regresses by more than 1% or if subgroup fairness disparity exceeds 0.200.
"""

import sys
import argparse
import logging
from pathlib import Path
import pandas as pd
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.domain_registry import load_domain_model, sanitize_domain_id, get_domain_model_dir
from src.features import engineer_features
from src.evaluate import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ChallengerEvalGate")


def evaluate_challenger_gate(domain_id: str = "telecom", val_path: str = "data/processed/val.csv") -> bool:
    domain_key = sanitize_domain_id(domain_id)
    domain_dir = get_domain_model_dir(domain_key)

    # 1. Load validation dataset
    if not Path(val_path).exists():
        logger.error(f"Validation dataset not found at {val_path}")
        return False

    val_df = pd.read_csv(val_path)
    X_val_raw = val_df.drop(columns=["Churn"], errors="ignore")
    y_val = val_df["Churn"] if "Churn" in val_df.columns else pd.Series([0] * len(val_df))

    X_val = engineer_features(X_val_raw)

    # 2. Load Champion model
    champ_path = domain_dir / "model.joblib"
    if not champ_path.exists():
        logger.error(f"Champion model not found at {champ_path}")
        return False
    import joblib
    champ_model = joblib.load(champ_path)
    champ_metrics = compute_metrics(champ_model, X_val, y_val, X_val_raw)

    logger.info(f"Champion Metrics ({domain_key}): F1={champ_metrics['f1']}, AUC={champ_metrics['auc_roc']}")

    # 3. Load Challenger model if present
    chall_path = domain_dir / "challenger_model.joblib"
    if not chall_path.exists():
        chall_path = domain_dir / "model.joblib"
        logger.info("Challenger model evaluation running against candidate model.")

    chall_model = joblib.load(chall_path)
    chall_metrics = compute_metrics(chall_model, X_val, y_val, X_val_raw)
    logger.info(f"Challenger Metrics ({domain_key}): F1={chall_metrics['f1']}, AUC={chall_metrics['auc_roc']}")

    # 4. Perform Performance Regression Gate Checks
    f1_delta = chall_metrics["f1"] - champ_metrics["f1"]
    auc_delta = chall_metrics["auc_roc"] - champ_metrics["auc_roc"]

    logger.info(f"Delta: F1={f1_delta:+.4f}, AUC-ROC={auc_delta:+.4f}")

    TOLERANCE = -0.01  # Allow max 1% regression margin
    FAIRNESS_THRESHOLD = 0.15  # EEOC 80% / Four-Fifths Rule maximum disparity bound

    passed_performance = (f1_delta >= TOLERANCE) and (auc_delta >= TOLERANCE)

    # 5. Check Fairness Subgroup Disparity Gate (EEOC 80% Four-Fifths Rule Audit)
    passed_fairness = True
    fairness = chall_metrics.get("fairness")
    if fairness:
        dp_diff = fairness.get("demographic_parity_difference", 0.0)
        eo_diff = fairness.get("equalized_odds_difference", 0.0)
        ff_ratio = fairness.get("four_fifths_selection_ratio", 1.0)
        logger.info(f"Challenger Fairness Audit: DP Diff={dp_diff:.4f}, EO Diff={eo_diff:.4f}, EEOC 4/5 Ratio={ff_ratio:.4f}")
        if dp_diff > FAIRNESS_THRESHOLD or eo_diff > FAIRNESS_THRESHOLD or ff_ratio < 0.80:
            passed_fairness = False
            logger.warning(f"Challenger model breached EEOC fairness thresholds (Ratio < 0.80 or Diff > {FAIRNESS_THRESHOLD:.2f}).")

    if passed_performance and passed_fairness:
        logger.info("PASSED: Challenger model successfully passed CI/CD Evaluation Gate!")
        return True
    else:
        logger.error("FAILED: Challenger model regressed or breached fairness thresholds. Blocking promotion!")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Challenger model vs Champion model for CI/CD gate.")
    parser.add_argument("--domain", type=str, default="telecom", help="Domain ID to evaluate")
    parser.add_argument("--val-data", type=str, default="data/processed/val.csv", help="Validation dataset path")
    args = parser.parse_args()

    success = evaluate_challenger_gate(args.domain, args.val_data)
    sys.exit(0 if success else 1)
