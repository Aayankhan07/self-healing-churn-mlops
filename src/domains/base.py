"""
Domain specifications.

A domain is a tenant of the platform: telecom, school, ecommerce, and any
custom domain bootstrapped at runtime. Before this module a "domain" was only a
folder name under models/ — every request was validated, healed, and feature-
engineered against the Telecom schema regardless of which domain it named.

A DomainSpec makes the contract explicit: which fields a domain accepts, how a
dirty value is repaired, and where its labels come from. The healing engine and
the request validator are both driven by the spec, so adding a domain is data,
not code.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# Rounding used when a repaired value is derived from other fields.
_DERIVED_PRECISION = 2


@dataclass(frozen=True)
class NumericRule:
    """
    Repair rule for a single numeric field.

    `minimum` clamps, `default` fills a missing or uncoercible value, and
    `derive` computes a replacement from the already-healed row when no static
    default makes sense (e.g. TotalCharges from MonthlyCharges * tenure).
    """

    name: str
    cast: type = float
    minimum: Optional[float] = None
    # Value substituted when the field is absent or cannot be coerced.
    default: Optional[float] = None
    derive: Optional[Callable[[Dict[str, Any]], float]] = None
    # Clamping a value at the boundary is wrong for a strictly-positive field;
    # `minimum_exclusive` substitutes `minimum` instead of allowing equality.
    minimum_exclusive: bool = False
    # Message fragments, kept per-rule so wording stays stable across domains.
    label_missing: Optional[str] = None
    label_coerced: Optional[str] = None
    label_invalid: Optional[str] = None
    label_clamped: Optional[str] = None

    def default_for(self, row: Dict[str, Any]) -> float:
        if self.derive is not None:
            return self.derive(row)
        return self.cast(self.default if self.default is not None else 0)


@dataclass(frozen=True)
class CategoricalRule:
    """
    Repair rule for a categorical field.

    Values are matched exactly, then by fuzzy match against `options` (to absorb
    typos), and finally replaced by the first option as a documented default.
    `aliases` maps known spellings that fuzzy matching would get wrong.
    """

    name: str
    options: List[str]
    aliases: Dict[str, str] = field(default_factory=dict)
    fuzzy_cutoff: float = 0.6

    @property
    def fallback(self) -> str:
        return self.options[0]

    def match(self, value: str) -> Optional[str]:
        """Return the canonical option for `value`, or None if unrecognized."""
        if value in self.options:
            return value
        alias = self.aliases.get(value.strip().lower())
        if alias:
            return alias
        close = difflib.get_close_matches(
            value, self.options, n=1, cutoff=self.fuzzy_cutoff
        )
        return close[0] if close else None


@dataclass(frozen=True)
class BinaryRule:
    """
    Repair rule for a field normalized to 0/1 (e.g. SeniorCitizen).

    Anything in `truthy` becomes 1; everything else becomes 0. The "already
    canonical" spellings are recorded so healing an input that is already clean
    reports no action.
    """

    name: str
    truthy: List[str] = field(
        default_factory=lambda: ["yes", "y", "true", "1", "1.0"]
    )
    canonical: List[str] = field(default_factory=lambda: ["0", "1"])


@dataclass(frozen=True)
class Constraint:
    """
    A cross-field invariant, checked after individual fields are repaired.

    `applies` decides whether the invariant is in force for a row, `holds`
    tests it, and `repair` returns the corrected row.
    """

    name: str
    applies: Callable[[Dict[str, Any]], bool]
    holds: Callable[[Dict[str, Any]], bool]
    repair: Callable[[Dict[str, Any]], Dict[str, Any]]
    message: str


@dataclass(frozen=True)
class RiskBands:
    """Probability cutoffs for the Low/Medium/High tiers."""

    low: float = 0.35
    high: float = 0.65

    def tier(self, probability: float) -> str:
        if probability >= self.high:
            return "High"
        if probability >= self.low:
            return "Medium"
        return "Low"


@dataclass(frozen=True)
class DomainSpec:
    """Everything the platform needs to know to serve one domain."""

    key: str
    display_name: str
    numeric: List[NumericRule] = field(default_factory=list)
    categorical: List[CategoricalRule] = field(default_factory=list)
    binary: List[BinaryRule] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    risk_bands: RiskBands = field(default_factory=RiskBands)
    target_column: str = "Churn"
    id_column: str = "customerID"
    # Source of ground-truth labels for retraining. None means the domain has no
    # labelled history, which callers must treat as "do not retrain".
    label_source_path: Optional[str] = None
    feature_engineering: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None

    @property
    def field_names(self) -> List[str]:
        names = [r.name for r in self.numeric]
        names += [r.name for r in self.binary]
        names += [r.name for r in self.categorical]
        return names

    def engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_engineering is None:
            return df
        return self.feature_engineering(df)


def round_derived(value: float) -> float:
    return round(value, _DERIVED_PRECISION)
