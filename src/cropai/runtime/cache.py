"""Process-local model cache. Dummy backends are cheap; still cache the pipeline."""

from __future__ import annotations

from typing import Any

_CACHE: dict[str, Any] = {}


def get_cached(key: str, factory):
    if key not in _CACHE:
        _CACHE[key] = factory()
    return _CACHE[key]


def clear_cache() -> None:
    _CACHE.clear()
