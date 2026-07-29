"""Unit tests for feature engineering and preprocessor pipeline."""

from src.features import engineer_features, build_preprocessor


def test_engineer_features(sample_dataframe):
    df_engineered = engineer_features(sample_dataframe)
    assert "tenure_group" in df_engineered.columns
    assert "services_count" in df_engineered.columns
    assert "charges_per_month_ratio" in df_engineered.columns
    assert "streaming_bundle_score" in df_engineered.columns
    assert "security_protection_score" in df_engineered.columns
    assert "is_month_to_month" in df_engineered.columns


def test_preprocessor_pipeline(sample_dataframe):
    df_engineered = engineer_features(sample_dataframe)
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(df_engineered)
    assert X_trans.shape[0] == len(sample_dataframe)
    assert X_trans.shape[1] > 0
