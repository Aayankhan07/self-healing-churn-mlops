"""
Spec-driven ingestion self-healing.

Repairs a dirty inbound record into one that satisfies its domain's schema,
returning the healed record alongside a human-readable list of what was
changed. The action strings are surfaced to API callers and written to the
self-healing log, so they are part of the observable contract — see
tests/test_characterization.py for the golden table that pins them.

Order matters and mirrors the original implementation: numeric fields first (so
derived defaults and cross-field constraints can rely on them), then binary
normalization, then categoricals, then constraints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.domains.base import (
    BinaryRule,
    CategoricalRule,
    DomainSpec,
    NumericRule,
)


def _heal_numeric(rule: NumericRule, row: Dict[str, Any], actions: List[str]) -> None:
    missing = rule.name not in row or row[rule.name] is None

    if missing:
        row[rule.name] = rule.cast(rule.default_for(row))
        if rule.label_missing:
            actions.append(rule.label_missing)
    elif not isinstance(row[rule.name], (int, float)) or isinstance(
        row[rule.name], bool
    ):
        try:
            row[rule.name] = rule.cast(float(row[rule.name]))
            if rule.label_coerced:
                actions.append(rule.label_coerced)
        except (TypeError, ValueError):
            row[rule.name] = rule.cast(rule.default_for(row))
            if rule.label_invalid:
                actions.append(rule.label_invalid)

    if rule.minimum is None:
        return

    value = row[rule.name]
    # For a strictly-positive field the boundary itself is invalid, so zero and
    # everything below it is replaced by the minimum rather than clamped to it.
    below = value <= 0 if rule.minimum_exclusive else value < rule.minimum
    if below:
        row[rule.name] = rule.cast(rule.minimum)
        if rule.label_clamped:
            actions.append(rule.label_clamped)


def _heal_binary(rule: BinaryRule, row: Dict[str, Any], actions: List[str]) -> None:
    if rule.name not in row or row[rule.name] is None:
        row[rule.name] = 0
        actions.append(f"Imputed missing {rule.name} to 0")
        return

    raw = str(row[rule.name]).strip().lower()
    normalized = 1 if raw in rule.truthy else 0
    row[rule.name] = normalized
    # Only report a change when the input was not already canonical.
    if raw != str(normalized):
        actions.append(f"Normalized {rule.name} to {normalized}")


def _heal_categorical(
    rule: CategoricalRule, row: Dict[str, Any], actions: List[str]
) -> None:
    if rule.name not in row or row[rule.name] is None:
        row[rule.name] = rule.fallback
        actions.append(f"Imputed missing {rule.name} to default '{rule.fallback}'")
        return

    value = str(row[rule.name]).strip()
    if value in rule.options:
        row[rule.name] = value
        return

    matched = rule.match(value)
    if matched is not None:
        row[rule.name] = matched
        actions.append(f"Mapped typos in {rule.name} ('{value}') to '{matched}'")
    else:
        row[rule.name] = rule.fallback
        actions.append(
            f"Imputed unrecognized {rule.name} ('{value}') "
            f"to default '{rule.fallback}'"
        )


def heal(
    raw_data: Dict[str, Any], spec: DomainSpec
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Repair `raw_data` against `spec`.

    Returns the healed record and the list of repairs performed. The input is
    never mutated. An already-clean record yields an empty action list.
    """
    row = dict(raw_data)
    actions: List[str] = []

    for rule in spec.numeric:
        _heal_numeric(rule, row, actions)

    # Constraints run here, before the categorical pass, so the reported order
    # of actions matches what callers have always seen: cross-field numeric
    # repairs are attributed alongside the numeric fields they correct.
    for constraint in spec.constraints:
        if constraint.applies(row) and not constraint.holds(row):
            row = constraint.repair(row)
            actions.append(constraint.message)

    for rule in spec.binary:
        _heal_binary(rule, row, actions)

    for rule in spec.categorical:
        _heal_categorical(rule, row, actions)

    return row, actions
