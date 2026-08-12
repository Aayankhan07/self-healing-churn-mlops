"""
Generic domain specification, inferred from a domain's baseline data.

Domains other than telecom have no hand-written schema. Rather than validating
their inputs against the telecom columns — which is what the platform did
before, and why a school record was silently scored as a telecom customer — we
read the domain's own baseline CSV and derive its fields from it.

Inference is deliberately conservative:
  - numeric columns become NumericRule with the column median as the default
  - low-cardinality object columns become CategoricalRule over observed values
  - high-cardinality object columns are treated as free-text identifiers and
    left alone, since imputing them to a "default" is meaningless

A domain with no usable baseline gets an empty spec, which the validator treats
as "accept the record as-is": permissive, but never silently telecom.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .base import CategoricalRule, DomainSpec, NumericRule, RiskBands

logger = logging.getLogger(__name__)

# An object column with more distinct values than this is an identifier or free
# text, not a category worth imputing.
MAX_CATEGORY_CARDINALITY = 25

# Columns that describe the record rather than the subject of prediction.
NON_FEATURE_COLUMNS = {
    "Churn",
    "customerID",
    "studentID",
    "id",
    "prediction_id",
}


def _numeric_rule(column: str, series: pd.Series) -> NumericRule:
    median = float(series.median()) if series.notna().any() else 0.0
    is_integral = pd.api.types.is_integer_dtype(series)
    cast = int if is_integral else float
    default = int(median) if is_integral else round(median, 4)
    # Baseline data is observational; a feature that is never negative in the
    # baseline is treated as non-negative.
    minimum = 0.0 if series.min() >= 0 else None
    return NumericRule(
        name=column,
        cast=cast,
        minimum=minimum,
        default=default,
        label_missing=f"Imputed missing {column} to baseline median ({default})",
        label_coerced=f"Coerced {column} to {cast.__name__}",
        label_invalid=f"Imputed invalid {column} to baseline median ({default})",
        label_clamped=f"Clamped negative {column} to 0",
    )


def infer_spec(
    domain_key: str,
    display_name: Optional[str] = None,
    baseline_path: Optional[Path] = None,
) -> DomainSpec:
    """Build a DomainSpec for `domain_key` from its baseline data."""
    numeric = []
    categorical = []

    if baseline_path is not None and Path(baseline_path).exists():
        try:
            frame = pd.read_csv(baseline_path)
        except Exception as exc:  # a malformed baseline must not break serving
            logger.warning(
                "Could not read baseline for domain '%s' (%s); "
                "falling back to a permissive schema",
                domain_key,
                exc,
            )
            frame = pd.DataFrame()

        for column in frame.columns:
            if column in NON_FEATURE_COLUMNS:
                continue
            series = frame[column].dropna()
            if series.empty:
                continue
            if pd.api.types.is_numeric_dtype(series):
                numeric.append(_numeric_rule(column, series))
                continue
            options = sorted(str(v) for v in series.unique())
            if len(options) <= MAX_CATEGORY_CARDINALITY:
                categorical.append(CategoricalRule(column, options))
    else:
        logger.info(
            "No baseline available for domain '%s'; using a permissive schema",
            domain_key,
        )

    return DomainSpec(
        key=domain_key,
        display_name=display_name or domain_key,
        numeric=numeric,
        categorical=categorical,
        binary=[],
        constraints=[],
        risk_bands=RiskBands(),
        # A generic domain has no curated ground-truth file. Retraining must
        # refuse rather than invent labels for it.
        label_source_path=None,
        feature_engineering=None,
    )
