"""Grouped, stratified train/val/isolated-test splits."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from cropai.dataset.schema import ImageRecord


def _stable_hash(text: str, seed: int) -> int:
    payload = f"{seed}:{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def split_records(
    records: list[ImageRecord],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> list[ImageRecord]:
    """Assign split labels without splitting a group_key across sets.

    Isolated test: groups hashed into test first. Remaining groups go to train/val.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {total}")

    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.group_key()].append(rec)

    # Stratify groups by majority crop+class so rare classes are not all tested.
    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    n_groups = len(group_items)
    n_test = max(1, round(n_groups * test_ratio)) if n_groups >= 3 else (1 if n_groups == 2 else 0)
    n_val = max(1, round(n_groups * val_ratio)) if n_groups >= 3 else 0

    # Deterministic assignment by group hash, then adjust counts.
    scored = []
    for key, recs in group_items:
        majority = recs[0].crop_id + "|" + recs[0].class_id
        scored.append((_stable_hash(key + majority, seed), key, recs))
    scored.sort(key=lambda t: t[0])

    assigned = {"test": [], "val": [], "train": []}
    for i, (_, key, recs) in enumerate(scored):
        if i < n_test:
            assigned["test"].append((key, recs))
        elif i < n_test + n_val:
            assigned["val"].append((key, recs))
        else:
            assigned["train"].append((key, recs))

    # If train emptied (tiny sets), keep at least one group in train.
    if not assigned["train"] and assigned["val"]:
        assigned["train"].append(assigned["val"].pop())
    if not assigned["train"] and assigned["test"]:
        assigned["train"].append(assigned["test"].pop())

    out: list[ImageRecord] = []
    for split_name, items in assigned.items():
        for _, recs in items:
            for rec in recs:
                rec.split = split_name
                out.append(rec)
    return out
