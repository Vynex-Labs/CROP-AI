"""Train/val/test leakage detection."""

from __future__ import annotations

from collections import defaultdict

from cropai.dataset.schema import ImageRecord


def detect_group_leakage(records: list[ImageRecord]) -> list[str]:
    """A group key must never appear in more than one split (except unassigned)."""
    group_splits: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        split = rec.split or "unassigned"
        if split == "unassigned":
            continue
        group_splits[rec.group_key()].add(split)
    leaks = []
    for group, splits in group_splits.items():
        if len(splits) > 1:
            leaks.append(f"group {group} appears in splits {sorted(splits)}")
    return leaks


def detect_sample_id_collision(records: list[ImageRecord]) -> list[str]:
    seen: dict[str, int] = {}
    errors = []
    for rec in records:
        seen[rec.sample_id] = seen.get(rec.sample_id, 0) + 1
    for sample_id, count in seen.items():
        if count > 1:
            errors.append(f"duplicate sample_id {sample_id} x{count}")
    return errors


def detect_hash_leakage(records: list[ImageRecord]) -> list[str]:
    """Identical file hashes must not cross splits."""
    hash_splits: dict[str, set[str]] = defaultdict(set)
    hash_ids: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        if not rec.sha256:
            continue
        hash_ids[rec.sha256].append(rec.sample_id)
        if rec.split and rec.split != "unassigned":
            hash_splits[rec.sha256].add(rec.split)
    errors = []
    for digest, splits in hash_splits.items():
        if len(splits) > 1:
            errors.append(
                f"sha256 {digest[:12]}… in splits {sorted(splits)} ids={hash_ids[digest]}"
            )
    return errors


def detect_relpath_collision(records: list[ImageRecord]) -> list[str]:
    seen: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        seen[rec.relpath].append(rec.sample_id)
    return [
        f"relpath {path} used by {ids}"
        for path, ids in seen.items()
        if len(ids) > 1
    ]
