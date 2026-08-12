"""
Equivalence tests between the spec-driven healing engine and the telecom rules
it replaced.

The healing action strings are part of the observable API contract, so the
extraction has to be behavior-preserving for telecom. These tests fuzz the two
implementations against each other rather than trusting a handful of examples.
"""

import random

import pytest

from src.domains.telecom import TELECOM_SPEC
from src.healing import heal

# Values chosen to exercise every branch: missing, wrong type, out of range,
# fuzzy-matchable typo, and unrecognized.
FUZZ_FIELDS = {
    "tenure": [-9, 0, 5, "3", "x", None, 72, 3.7],
    "MonthlyCharges": [None, 0, -2, "55.5", 70.35, 19.9, "bad"],
    "TotalCharges": [None, -4, "900", 0, 1500.0, "zz"],
    "SeniorCitizen": [None, 0, 1, "1", "0", "Yes", "no", "TRUE", "maybe", 2],
    "gender": [None, "Male", "female", "MALE", "Femalee", "x"],
    "Contract": [None, "Month-to-month", "One year", "Two yr", "Weekly"],
    "InternetService": [None, "DSL", "Fibre optic", "Fiber optic", "No", "none"],
    "PaymentMethod": [
        None,
        "Electronic check",
        "Electrnic check",
        "Mailed chek",
        "cash",
    ],
    "PhoneService": [None, "Yes", "No", "yep"],
    "MultipleLines": [None, "Yes", "No", "No phone service", "No phone svc"],
    "Partner": [None, "Yes", "No", "yes"],
    "Dependents": [None, "Yes", "No"],
    "OnlineSecurity": [None, "Yes", "No", "No internet service"],
    "StreamingTV": [None, "Yes", "No"],
    "PaperlessBilling": [None, "Yes", "No"],
}


def _random_record(rng):
    record = {}
    for field, values in FUZZ_FIELDS.items():
        if rng.random() < 0.75:
            value = rng.choice(values)
            # Keep some explicit Nones so the "present but null" branch is hit.
            if value is not None or rng.random() < 0.3:
                record[field] = value
    return record


def test_matches_legacy_telecom_healing_under_fuzzing():
    """
    The extracted engine must reproduce the legacy rules exactly — same healed
    values, same action strings, same order.
    """
    from tests.legacy_healing_reference import heal_customer_data as legacy_heal

    rng = random.Random(99)
    mismatches = []

    for _ in range(2000):
        record = _random_record(rng)
        legacy_data, legacy_actions = legacy_heal(dict(record))
        spec_data, spec_actions = heal(dict(record), TELECOM_SPEC)
        if legacy_data != spec_data or legacy_actions != spec_actions:
            mismatches.append((record, legacy_actions, spec_actions))

    assert not mismatches, f"{len(mismatches)} divergences, first: {mismatches[0]}"


@pytest.mark.parametrize("raw,expected", [(True, 1), (False, 0)])
def test_boolean_tenure_is_coerced(raw, expected):
    """
    A deliberate improvement over the legacy rules.

    Python's bool is a subclass of int, so the legacy isinstance check let a
    raw `True` through untouched and it reached the model as a tenure value.
    The spec engine coerces it and reports the repair.
    """
    healed, actions = heal({"tenure": raw}, TELECOM_SPEC)
    assert healed["tenure"] == expected
    assert "Coerced tenure to integer" in actions


def test_heal_is_pure():
    record = {"tenure": -5, "Contract": "Weekly"}
    snapshot = dict(record)
    heal(record, TELECOM_SPEC)
    assert record == snapshot
