"""
Per-domain schema tests.

Before Phase 2 every domain was validated and healed against the Telecom
schema, so a school record was silently repaired into a telecom customer. These
tests assert each domain now answers for its own fields.
"""

import pandas as pd
import pytest

from src.domains import get_domain_spec, reset_spec_cache
from src.domains.base import CategoricalRule, NumericRule, RiskBands
from src.domains.generic import infer_spec
from src.healing import heal


def test_telecom_spec_is_the_handwritten_one():
    spec = get_domain_spec("telecom")
    assert spec.key == "telecom"
    assert spec.feature_engineering is not None
    assert spec.label_source_path is not None
    assert {r.name for r in spec.numeric} == {
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    }
    assert "SeniorCitizen" in {r.name for r in spec.binary}


def test_display_names_resolve_to_the_same_spec():
    """The dashboard sends human-readable names; the API sends keys."""
    assert get_domain_spec("Telecom Customer Churn").key == "telecom"
    assert get_domain_spec("telecom").key == "telecom"


def test_school_spec_carries_its_own_fields():
    """
    The school baseline has genuine domain columns. They must appear in its
    spec, which is exactly what the telecom-only schema could never express.
    """
    spec = get_domain_spec("school")
    fields = set(spec.field_names)
    assert "attendance_percentage" in fields
    assert "gpa_average" in fields
    assert "grade_level" in fields


def test_school_heals_its_own_numeric_fields():
    spec = get_domain_spec("school")
    healed, actions = heal({"gpa_average": "3.2"}, spec)
    assert healed["gpa_average"] == pytest.approx(3.2)
    assert any("gpa_average" in a for a in actions)


def test_generic_domain_has_no_label_source():
    """
    Retraining a domain with no ground truth would train it on its own
    predictions, so a generic spec must not claim a label source.
    """
    for domain in ["school", "ecommerce", "fitness"]:
        assert get_domain_spec(domain).label_source_path is None


def test_unknown_domain_gets_a_permissive_spec(tmp_path):
    spec = infer_spec("custom_nothing", baseline_path=tmp_path / "missing.csv")
    assert spec.key == "custom_nothing"
    assert spec.numeric == []
    assert spec.categorical == []
    # A permissive spec passes records through rather than telecom-ising them.
    healed, actions = heal({"anything": 1}, spec)
    assert healed == {"anything": 1}
    assert actions == []


def test_inference_reads_types_from_the_baseline(tmp_path):
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame(
        {
            "visits": [1, 4, 9],
            "score": [0.5, 0.25, 0.75],
            "plan": ["basic", "pro", "basic"],
            "customerID": ["a", "b", "c"],
            "Churn": [0, 1, 0],
        }
    ).to_csv(baseline, index=False)

    spec = infer_spec("custom_gym", baseline_path=baseline)

    numeric = {r.name: r for r in spec.numeric}
    categorical = {r.name: r for r in spec.categorical}

    assert set(numeric) == {"visits", "score"}
    assert numeric["visits"].cast is int
    assert numeric["score"].cast is float
    assert set(categorical) == {"plan"}
    assert categorical["plan"].options == ["basic", "pro"]
    # Identifier and target columns are not features.
    assert "customerID" not in numeric and "customerID" not in categorical
    assert "Churn" not in numeric and "Churn" not in categorical


def test_high_cardinality_columns_are_not_treated_as_categories(tmp_path):
    baseline = tmp_path / "baseline.csv"
    pd.DataFrame({"note": [f"free text {i}" for i in range(60)]}).to_csv(
        baseline, index=False
    )
    spec = infer_spec("custom_notes", baseline_path=baseline)
    assert spec.categorical == []


def test_malformed_baseline_falls_back_to_permissive(tmp_path):
    baseline = tmp_path / "broken.csv"
    baseline.write_bytes(b"\x00\x01\x02 not,a,valid\ncsv")
    spec = infer_spec("custom_broken", baseline_path=baseline)
    assert spec.key == "custom_broken"


def test_spec_cache_returns_a_stable_object():
    reset_spec_cache()
    first = get_domain_spec("ecommerce")
    second = get_domain_spec("ecommerce")
    assert first is second


def test_reset_spec_cache_forces_reinference():
    reset_spec_cache()
    first = get_domain_spec("fitness")
    reset_spec_cache()
    assert get_domain_spec("fitness") is not first


# ── Risk bands ──────────────────────────────────────────────


def test_risk_bands_map_probabilities_to_tiers():
    bands = RiskBands(low=0.35, high=0.65)
    assert bands.tier(0.0) == "Low"
    assert bands.tier(0.34) == "Low"
    assert bands.tier(0.35) == "Medium"
    assert bands.tier(0.64) == "Medium"
    assert bands.tier(0.65) == "High"
    assert bands.tier(1.0) == "High"


def test_domains_may_set_their_own_bands():
    bands = RiskBands(low=0.2, high=0.5)
    assert bands.tier(0.25) == "Medium"
    assert RiskBands().tier(0.25) == "Low"


# ── Rule primitives ─────────────────────────────────────────


def test_categorical_rule_matches_exact_then_fuzzy_then_gives_up():
    rule = CategoricalRule("Contract", ["Month-to-month", "One year", "Two year"])
    assert rule.match("One year") == "One year"
    assert rule.match("One yr") == "One year"
    assert rule.match("Weekly") is None
    assert rule.fallback == "Month-to-month"


def test_categorical_aliases_beat_fuzzy_matching():
    rule = CategoricalRule("plan", ["Basic", "Premium"], aliases={"prem": "Premium"})
    assert rule.match("prem") == "Premium"


def test_numeric_rule_derives_when_no_static_default():
    rule = NumericRule(name="total", derive=lambda row: row["rate"] * 2)
    assert rule.default_for({"rate": 5}) == 10
