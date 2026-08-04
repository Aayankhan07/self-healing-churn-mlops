import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

import os
import yaml
import logging
from src.domain_registry import ensure_domain_initialized, sanitize_domain_id
from src.train import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOMAINS = ["telecom", "school", "ecommerce", "fitness"]

def train_all():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    for domain in DOMAINS:
        logger.info(f"--- Training domain model for: {domain} ---")
        os.environ["TARGET_DOMAIN"] = domain
        
        if domain == "school":
            sample_school = BASE_DIR / "data" / "sample_batch_students.csv"
            if sample_school.exists():
                os.environ["TRAIN_DATA_PATH"] = str(sample_school)
            else:
                os.environ["TRAIN_DATA_PATH"] = "data/processed/train.csv"
        else:
            os.environ["TRAIN_DATA_PATH"] = "data/processed/train.csv"

        ensure_domain_initialized(domain)
        train(params)
        logger.info(f"Finished training for {domain}")

if __name__ == "__main__":
    train_all()
