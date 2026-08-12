"""
Frozen copy of the original telecom healing rules.

Kept solely as the reference implementation for
tests/test_healing_equivalence.py, which fuzzes the spec-driven engine in
src/healing.py against it to prove the extraction was behavior-preserving.

Do not call this from production code and do not "improve" it — its value is
that it still behaves exactly as the pre-refactor API did.
"""

import difflib


def heal_customer_data(raw_data: dict) -> tuple[dict, list[str]]:
    healed_data = raw_data.copy()
    healed_actions = []

    # Rules mapping
    # 1. Numeric fields: negative or missing values clamped or imputed
    # tenure: int
    if "tenure" not in healed_data or healed_data["tenure"] is None:
        healed_data["tenure"] = 0
        healed_actions.append("Imputed missing tenure to 0")
    elif not isinstance(healed_data["tenure"], (int, float)):
        try:
            healed_data["tenure"] = int(float(healed_data["tenure"]))
            healed_actions.append("Coerced tenure to integer")
        except Exception:
            healed_data["tenure"] = 0
            healed_actions.append("Imputed invalid tenure to 0")

    if healed_data["tenure"] < 0:
        healed_data["tenure"] = 0
        healed_actions.append("Clamped negative tenure to 0")

    # MonthlyCharges: float
    if "MonthlyCharges" not in healed_data or healed_data["MonthlyCharges"] is None:
        median_mc = 70.35
        healed_data["MonthlyCharges"] = median_mc
        healed_actions.append(f"Imputed missing MonthlyCharges to median ({median_mc})")
    elif not isinstance(healed_data["MonthlyCharges"], (int, float)):
        try:
            healed_data["MonthlyCharges"] = float(healed_data["MonthlyCharges"])
            healed_actions.append("Coerced MonthlyCharges to float")
        except Exception:
            median_mc = 70.35
            healed_data["MonthlyCharges"] = median_mc
            healed_actions.append(
                f"Imputed invalid MonthlyCharges to median ({median_mc})"
            )

    if healed_data["MonthlyCharges"] <= 0:
        healed_data["MonthlyCharges"] = 0.01
        healed_actions.append("Clamped non-positive MonthlyCharges to 0.01")

    # TotalCharges: float
    if "TotalCharges" not in healed_data or healed_data["TotalCharges"] is None:
        healed_data["TotalCharges"] = round(
            healed_data["MonthlyCharges"] * max(1, healed_data["tenure"]), 2
        )
        healed_actions.append("Imputed missing TotalCharges as MonthlyCharges * tenure")
    elif not isinstance(healed_data["TotalCharges"], (int, float)):
        try:
            healed_data["TotalCharges"] = float(healed_data["TotalCharges"])
            healed_actions.append("Coerced TotalCharges to float")
        except Exception:
            healed_data["TotalCharges"] = round(
                healed_data["MonthlyCharges"] * max(1, healed_data["tenure"]), 2
            )
            healed_actions.append(
                "Imputed invalid TotalCharges as MonthlyCharges * tenure"
            )

    if healed_data["TotalCharges"] < 0:
        healed_data["TotalCharges"] = 0.0
        healed_actions.append("Clamped negative TotalCharges to 0.0")

    # Constraint check: TotalCharges cannot be less than MonthlyCharges when tenure > 0
    if (
        healed_data["tenure"] > 0
        and healed_data["TotalCharges"] < healed_data["MonthlyCharges"]
    ):
        healed_data["TotalCharges"] = round(
            healed_data["MonthlyCharges"] * healed_data["tenure"], 2
        )
        healed_actions.append(
            "Recomputed TotalCharges as MonthlyCharges * tenure due to constraint mismatch"
        )

    # 2. SeniorCitizen: normalize to 0 or 1
    if "SeniorCitizen" not in healed_data or healed_data["SeniorCitizen"] is None:
        healed_data["SeniorCitizen"] = 0
        healed_actions.append("Imputed missing SeniorCitizen to 0")
    else:
        val = str(healed_data["SeniorCitizen"]).strip().lower()
        if val in ("yes", "y", "true", "1", "1.0"):
            healed_data["SeniorCitizen"] = 1
            if val != "1":
                healed_actions.append("Normalized SeniorCitizen to 1")
        else:
            healed_data["SeniorCitizen"] = 0
            if val != "0":
                healed_actions.append("Normalized SeniorCitizen to 0")

    # 3. Categorical features string similarity dynamic mapping
    CATEGORICAL_SCHEMAS = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["Yes", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    }

    for col, valid_options in CATEGORICAL_SCHEMAS.items():
        if col not in healed_data or healed_data[col] is None:
            default_cat = valid_options[0]
            healed_data[col] = default_cat
            healed_actions.append(f"Imputed missing {col} to default '{default_cat}'")
        else:
            val = str(healed_data[col]).strip()
            if val in valid_options:
                healed_data[col] = val
                continue

            # String similarity lookup
            matches = difflib.get_close_matches(val, valid_options, n=1, cutoff=0.6)
            if matches:
                closest = matches[0]
                healed_data[col] = closest
                healed_actions.append(f"Mapped typos in {col} ('{val}') to '{closest}'")
            else:
                default_cat = valid_options[0]
                healed_data[col] = default_cat
                healed_actions.append(
                    f"Imputed unrecognized {col} ('{val}') to default '{default_cat}'"
                )

    return healed_data, healed_actions
