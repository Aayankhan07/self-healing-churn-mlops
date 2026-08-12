"""
Domain spec lookup.

Resolves a domain id — however the caller spelled it — to the DomainSpec that
governs validation, healing, and risk banding for that domain. Telecom is
hand-written; everything else is inferred from its baseline data and cached,
since inference reads a CSV.
"""

from __future__ import annotations

import threading
from typing import Dict

from src.domain_registry import get_domain_baseline_path, sanitize_domain_id

from .base import DomainSpec
from .generic import infer_spec
from .telecom import TELECOM_SPEC

# Hand-written specs, keyed by sanitized domain id.
_EXPLICIT_SPECS: Dict[str, DomainSpec] = {
    TELECOM_SPEC.key: TELECOM_SPEC,
}

_inferred_cache: Dict[str, DomainSpec] = {}
_cache_lock = threading.Lock()


def get_domain_spec(domain_id: str) -> DomainSpec:
    """Return the DomainSpec for `domain_id`, inferring one if needed."""
    key = sanitize_domain_id(domain_id)

    explicit = _EXPLICIT_SPECS.get(key)
    if explicit is not None:
        return explicit

    with _cache_lock:
        cached = _inferred_cache.get(key)
        if cached is not None:
            return cached

    spec = infer_spec(key, baseline_path=get_domain_baseline_path(key))

    with _cache_lock:
        # Another thread may have inferred the same spec; either is equivalent.
        return _inferred_cache.setdefault(key, spec)


def reset_spec_cache() -> None:
    """Drop inferred specs. Used after a domain's baseline is rewritten."""
    with _cache_lock:
        _inferred_cache.clear()
