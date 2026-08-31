"""Pure-Python forecasting metrics. No invented numbers — only on provided labels."""

from __future__ import annotations

from typing import Sequence


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    pos = [s for y, s in zip(y_true, y_score, strict=False) if int(y) == 1]
    neg = [s for y, s in zip(y_true, y_score, strict=False) if int(y) == 0]
    if not pos or not neg:
        return None
    gt = eq = 0
    for p in pos:
        for n in neg:
            if p > n:
                gt += 1
            elif p == n:
                eq += 1
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def average_precision(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    pairs = sorted(zip(y_score, y_true, strict=False), key=lambda t: t[0], reverse=True)
    if not any(y == 1 for _, y in pairs) or not pairs:
        return None
    tp = 0
    fp = 0
    fn = sum(1 for _, y in pairs if int(y) == 1)
    ap = 0.0
    prev_rec = 0.0
    for score, y in pairs:
        if int(y) == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        ap += prec * (rec - prev_rec)
        prev_rec = rec
    return ap


def brier(y_true: Sequence[int], y_score: Sequence[float]) -> float | None:
    pairs = list(zip(y_true, y_score, strict=False))
    if not pairs:
        return None
    return sum((float(s) - float(y)) ** 2 for y, s in pairs) / len(pairs)


def false_negative_rate(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> float | None:
    fn = fp = tp = 0
    for y, s in zip(y_true, y_score, strict=False):
        pred = 1 if float(s) >= threshold else 0
        if int(y) == 1 and pred == 0:
            fn += 1
        elif int(y) == 1 and pred == 1:
            tp += 1
        elif int(y) == 0 and pred == 1:
            fp += 1
    denom = tp + fn
    return (fn / denom) if denom else None


def binary_prf(y_true: Sequence[int], y_score: Sequence[float], threshold: float = 0.5) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for y, s in zip(y_true, y_score, strict=False):
        pred = 1 if float(s) >= threshold else 0
        if int(y) == 1 and pred == 1:
            tp += 1
        elif int(y) == 0 and pred == 1:
            fp += 1
        elif int(y) == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def expected_calibration_error(
    y_true: Sequence[int], y_score: Sequence[float], n_bins: int = 10
) -> float | None:
    n = min(len(y_true), len(y_score))
    if n == 0:
        return None
    bins = [[] for _ in range(n_bins)]
    for i in range(n):
        s = min(0.999999, max(0.0, float(y_score[i])))
        b = min(n_bins - 1, int(s * n_bins))
        bins[b].append((int(y_true[i]), s))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        acc = sum(y for y, _ in bucket) / len(bucket)
        conf = sum(s for _, s in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(acc - conf)
    return ece


def report(y_true: Sequence[int], y_score: Sequence[float], *, source: str) -> dict:
    if not y_true:
        return {
            "roc_auc": "NOT MEASURED",
            "pr_auc": "NOT MEASURED",
            "brier": "NOT MEASURED",
            "fnr": "NOT MEASURED",
            "ece": "NOT MEASURED",
            "precision": "NOT MEASURED",
            "recall": "NOT MEASURED",
            "macro_f1": "NOT MEASURED",
            "lead_time": "NOT MEASURED",
            "n": 0,
            "source": source,
        }
    prf = binary_prf(y_true, y_score)
    return {
        "roc_auc": roc_auc(y_true, y_score),
        "pr_auc": average_precision(y_true, y_score),
        "brier": brier(y_true, y_score),
        "fnr": false_negative_rate(y_true, y_score),
        "ece": expected_calibration_error(y_true, y_score),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "macro_f1": prf["f1"],
        "lead_time": "NOT MEASURED",
        "n": len(list(y_true)),
        "source": source,
        "note": "Do not report synthetic_source metrics as outbreak skill."
        if source.startswith("synthetic")
        else "",
    }
