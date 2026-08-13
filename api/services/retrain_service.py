"""
Self-healing retraining.

Rebuilds a domain's model from its recent production inputs merged with its
training split, then registers the result as a challenger for shadow
evaluation. The champion keeps serving until someone promotes the challenger.

A domain with no ground-truth label source is refused: every label would be a
pseudo-label derived from the model's own output, which trains the model on its
own opinions.
"""

import json
import logging
import os
import random

import pandas as pd
import yaml

from api.services import model_registry
from src.domains import get_domain_spec
from src.healing import heal

logger = logging.getLogger(__name__)


def run_self_healing_retraining(app_ref, domain_id: str = "telecom"):
    """
    Background worker that runs the self-healing retraining loop for a specific domain.
    Reads recent production inputs, resolves labels, merges them with domain baseline data,
    triggers Optuna + XGBoost tuning, and atomically hot-reloads the new domain model into memory.
    """
    from api.database import SessionLocal, log_self_healing_event, last_n_inputs
    from src.train import train as run_training_pipeline
    from src.domain_registry import (
        load_domain_model,
        load_domain_preprocessor,
        sanitize_domain_id,
    )

    domain_id = sanitize_domain_id(domain_id)
    os.environ["TARGET_DOMAIN"] = domain_id

    db = SessionLocal()
    try:
        # 1. Fetch recent production inputs from SQLite, scoped to this domain
        records = last_n_inputs(db, n=500, domain_id=domain_id)
        if not records:
            log_self_healing_event(
                db,
                "retraining",
                f"Auto-retraining aborted for domain '{domain_id}': no prediction records found.",
                domain_id=domain_id,
            )
            model_registry.release_retraining_slot(app_ref)
            db.close()
            return

        # Resolve ground-truth labels from this domain's own label source. A
        # domain without one must not be retrained: every row would be a
        # pseudo-label derived from the model's own output, which trains the
        # model on its own opinions.
        spec = get_domain_spec(domain_id)
        raw_path = spec.label_source_path
        if not raw_path:
            log_self_healing_event(
                db,
                "retraining",
                f"Auto-retraining aborted for domain '{domain_id}': no ground-truth "
                f"label source is configured, so every label would be a pseudo-label.",
                domain_id=domain_id,
            )
            model_registry.release_retraining_slot(app_ref)
            db.close()
            return

        raw_labels = {}
        if os.path.exists(raw_path):
            raw_df = pd.read_csv(raw_path)
            id_col = spec.id_column
            target = spec.target_column
            if id_col in raw_df.columns and target in raw_df.columns:
                raw_df[id_col] = raw_df[id_col].astype(str).str.strip()
                raw_labels = (
                    raw_df.set_index(id_col)[target].map({"Yes": 1, "No": 0}).to_dict()
                )

        # Reconstruct DataFrame with labels
        new_rows = []
        sample_weights = []

        pseudo_label_count = 0
        true_label_count = 0

        for r in records:
            if not r.features_json:
                continue
            try:
                features = json.loads(r.features_json)
                cust_id = str(r.customer_id).strip() if r.customer_id else None

                features, db_healed_actions = heal(features, spec)
                if db_healed_actions:
                    log_self_healing_event(
                        db,
                        "data_quality",
                        f"Database-level self-healing corrected features for retraining of Customer {cust_id or 'N/A'}: {', '.join(db_healed_actions)}",
                        domain_id=domain_id,
                    )

                label = None
                weight = 1.0

                if cust_id and cust_id in raw_labels:
                    label = raw_labels[cust_id]
                    true_label_count += 1
                else:
                    # Enforce pseudo-label ratio cap (max 30% pseudo-labels) and weight decay (0.25)
                    MAX_PSEUDO_LABEL_RATIO = 0.30
                    current_pseudo_ratio = pseudo_label_count / max(
                        1, (true_label_count + pseudo_label_count)
                    )
                    if current_pseudo_ratio < MAX_PSEUDO_LABEL_RATIO:
                        if r.probability >= 0.85:
                            label = 1
                            weight = 0.25
                            pseudo_label_count += 1
                        elif r.probability <= 0.15:
                            label = 0
                            weight = 0.25
                            pseudo_label_count += 1

                if label is not None:
                    if weight == 0.25:
                        if random.random() > 0.5:
                            continue
                    features["Churn"] = label
                    new_rows.append(features)
                    sample_weights.append(weight)
            except Exception:
                pass

        if not new_rows:
            log_self_healing_event(
                db,
                "retraining",
                f"Auto-retraining aborted for domain '{domain_id}': no labeled records resolved.",
                domain_id=domain_id,
            )
            model_registry.release_retraining_slot(app_ref)
            db.close()
            return

        new_prod_df = pd.DataFrame(new_rows)
        train_path = "data/processed/train.csv"
        combined_path = "data/processed/train_retrain.csv"

        if not os.path.exists(train_path):
            model_registry.release_retraining_slot(app_ref)
            db.close()
            return

        original_train_df = pd.read_csv(train_path)
        combined_train_df = pd.concat(
            [original_train_df, new_prod_df], ignore_index=True
        )
        combined_train_df.to_csv(combined_path, index=False)
        os.environ["TRAIN_DATA_PATH"] = combined_path

        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        log_self_healing_event(
            db,
            "retraining",
            f"Starting retraining pipeline for domain '{domain_id}'. Combined dataset has {len(combined_train_df)} samples ({true_label_count} true labels, {pseudo_label_count} pseudo-labels).",
            domain_id=domain_id,
        )

        run_training_pipeline(params)

        if os.path.exists(combined_path):
            os.remove(combined_path)

        # Register retrained model as Challenger (Champion remains active until promoted)
        new_model = load_domain_model(domain_id)
        new_preprocessor = load_domain_preprocessor(domain_id)

        model_registry.register_challenger(
            app_ref,
            domain_id,
            new_model,
            new_preprocessor,
            f"{domain_id}-challenger-v2",
        )

        log_self_healing_event(
            db,
            "retraining",
            f"Auto-retraining completed for domain '{domain_id}'. Retrained model registered as Challenger in shadow evaluation mode.",
            domain_id=domain_id,
        )
        from src.notifications import send_slack_alert

        send_slack_alert(
            "retraining",
            f"Challenger Model Registered for domain {domain_id}",
            "Auto-retraining completed. Challenger version registered for shadow deployment evaluation.",
            domain_id=domain_id,
        )
    except Exception as e:
        log_self_healing_event(
            db,
            "retraining",
            f"Auto-retraining failed: {str(e)}",
            domain_id=domain_id,
        )
        # Clean up combined path on error
        combined_path = "data/processed/train_retrain.csv"
        if os.path.exists(combined_path):
            try:
                os.remove(combined_path)
            except Exception:
                pass
    finally:
        model_registry.release_retraining_slot(app_ref)
        db.close()
