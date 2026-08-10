import pandas as pd
from src.domain_registry import (
    sanitize_domain_id,
    get_domain_model_dir,
    get_domain_baseline_path,
    ensure_domain_initialized,
    load_domain_model,
    load_domain_preprocessor,
    bootstrap_custom_domain,
)


def test_sanitize_domain_id():
    assert sanitize_domain_id("Telecom Customer Churn") == "telecom"
    assert sanitize_domain_id("School Student Churn") == "school"
    assert (
        sanitize_domain_id("Hospital Patient Readmission")
        == "custom_hospital_patient_readmission"
    )


def test_domain_initialization_and_loading():
    ensure_domain_initialized("school")
    model_dir = get_domain_model_dir("school")
    assert (model_dir / "model.joblib").exists()
    assert (model_dir / "preprocessor.joblib").exists()

    baseline_path = get_domain_baseline_path("school")
    assert baseline_path.exists()

    model = load_domain_model("school")
    assert model is not None
    prep = load_domain_preprocessor("school")
    assert prep is not None


def test_bootstrap_custom_domain():
    df_sample = pd.DataFrame({"colA": [1, 2], "colB": [3, 4]})
    key = bootstrap_custom_domain("Hospital Patient Readmission", df_sample)
    assert key == "custom_hospital_patient_readmission"

    baseline_path = get_domain_baseline_path(key)
    assert baseline_path.exists()
    saved_df = pd.read_csv(baseline_path)
    assert len(saved_df) == 2
