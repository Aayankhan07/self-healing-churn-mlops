"""Per-domain schema specifications."""

from .base import (
    BinaryRule,
    CategoricalRule,
    Constraint,
    DomainSpec,
    NumericRule,
    RiskBands,
)
from .registry import get_domain_spec, reset_spec_cache
from .telecom import TELECOM_SPEC

__all__ = [
    "BinaryRule",
    "CategoricalRule",
    "Constraint",
    "DomainSpec",
    "NumericRule",
    "RiskBands",
    "TELECOM_SPEC",
    "get_domain_spec",
    "reset_spec_cache",
]
