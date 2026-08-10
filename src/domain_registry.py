import shutil
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
BASELINES_DIR = BASE_DIR / "data" / "baselines"

# Map human domain names to filesystem keys
DOMAIN_KEY_MAP = {
    "Telecom Customer Churn": "telecom",
    "School Student Churn": "school",
    "E-Commerce Customer Churn": "ecommerce",
    "Fitness Club Member Churn": "fitness",
}


def sanitize_domain_id(domain_name: str) -> str:
    """Convert domain name into a clean filesystem folder key."""
    if domain_name in DOMAIN_KEY_MAP:
        return DOMAIN_KEY_MAP[domain_name]
    clean_name = domain_name.lower().replace(" ", "_").replace("-", "_")
    clean_name = "".join(c for c in clean_name if c.isalnum() or c == "_")
    if (
        not clean_name.startswith("custom_")
        and clean_name not in DOMAIN_KEY_MAP.values()
    ):
        clean_name = f"custom_{clean_name}"
    return clean_name


def get_domain_model_dir(domain_id: str) -> Path:
    domain_id = sanitize_domain_id(domain_id)
    domain_dir = MODELS_DIR / domain_id
    domain_dir.mkdir(parents=True, exist_ok=True)
    return domain_dir


def get_domain_baseline_path(domain_id: str) -> Path:
    domain_id = sanitize_domain_id(domain_id)
    BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    return BASELINES_DIR / f"{domain_id}_baseline.csv"


def ensure_domain_initialized(domain_id: str):
    """Ensure a domain has isolated model, preprocessor, and baseline artifacts."""
    domain_key = sanitize_domain_id(domain_id)
    domain_dir = get_domain_model_dir(domain_key)
    model_path = domain_dir / "model.joblib"
    prep_path = domain_dir / "preprocessor.joblib"
    baseline_path = get_domain_baseline_path(domain_key)

    # Base fallback paths
    default_model = MODELS_DIR / "model.joblib"
    if not default_model.exists():
        default_model = MODELS_DIR / "telecom" / "model.joblib"

    default_prep = MODELS_DIR / "preprocessor.joblib"
    if not default_prep.exists():
        default_prep = MODELS_DIR / "telecom" / "preprocessor.joblib"

    default_test_df = BASE_DIR / "data" / "processed" / "train.csv"

    if not model_path.exists() and default_model.exists():
        shutil.copy(default_model, model_path)

    if not prep_path.exists() and default_prep.exists():
        shutil.copy(default_prep, prep_path)

    if not baseline_path.exists():
        if domain_key == "school":
            sample_school = BASE_DIR / "data" / "sample_batch_students.csv"
            if sample_school.exists():
                shutil.copy(sample_school, baseline_path)
            elif default_test_df.exists():
                shutil.copy(default_test_df, baseline_path)
        elif default_test_df.exists():
            shutil.copy(default_test_df, baseline_path)


def load_domain_model(domain_id: str):
    ensure_domain_initialized(domain_id)
    domain_dir = get_domain_model_dir(domain_id)
    return joblib.load(domain_dir / "model.joblib")


def load_domain_preprocessor(domain_id: str):
    ensure_domain_initialized(domain_id)
    domain_dir = get_domain_model_dir(domain_id)
    return joblib.load(domain_dir / "preprocessor.joblib")


def bootstrap_custom_domain(domain_name: str, baseline_df: pd.DataFrame = None) -> str:
    """Bootstrap isolated artifacts for a brand new custom domain."""
    domain_key = sanitize_domain_id(domain_name)
    ensure_domain_initialized(domain_key)

    if baseline_df is not None and not baseline_df.empty:
        baseline_path = get_domain_baseline_path(domain_key)
        baseline_df.to_csv(baseline_path, index=False)

    return domain_key
