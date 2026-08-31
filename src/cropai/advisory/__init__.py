"""Structured IPM advisory and expert-validation engine (Phase 4)."""

from cropai.advisory.engine import AdvisoryEngine, advise
from cropai.advisory.schema import AdvisoryOutput, AdvisoryRequest

__all__ = ["AdvisoryEngine", "AdvisoryOutput", "AdvisoryRequest", "advise"]
