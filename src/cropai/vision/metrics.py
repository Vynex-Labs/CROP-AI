"""Pure-Python metrics. No fabricated scores — only compute what is passed in."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Sequence


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred, strict=False)) / len(y_true)


def _prf(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, float]]:
    labels = sorted(set(y_true) | set(y_pred))
    out: dict[str, dict[str, float]] = {}
    for lab in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == lab and p == lab)
        fp = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t != lab and p == lab)
        fn = sum(1 for t, p in zip(y_true, y_pred, strict=False) if t == lab and p != lab)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": float(tp + fn)}
    return out


def classification_report(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    per = _prf(y_true, y_pred)
    if not per:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "per_class": {},
            "confusion": {},
        }
    n = len(per)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred, strict=False):
        confusion[t][p] += 1
    return {
        "accuracy": accuracy(y_true, y_pred),
        "macro_precision": sum(v["precision"] for v in per.values()) / n,
        "macro_recall": sum(v["recall"] for v in per.values()) / n,
        "macro_f1": sum(v["f1"] for v in per.values()) / n,
        "per_class": per,
        "confusion": {t: dict(preds) for t, preds in confusion.items()},
        "n": len(y_true),
    }


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> dict[str, dict[str, int]]:
    return classification_report(y_true, y_pred)["confusion"]


def iou_box(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def detection_prf(
    gt_boxes: Iterable[tuple[str, Sequence[float]]],
    pred_boxes: Iterable[tuple[str, Sequence[float]]],
    iou_thresh: float = 0.5,
) -> dict[str, float]:
    """Greedy one-to-one matching at a single IoU threshold (not COCO mAP)."""
    gt = list(gt_boxes)
    pred = list(pred_boxes)
    used = set()
    tp = 0
    for cls, box in pred:
        best_i = None
        best_iou = iou_thresh
        for i, (gcls, gbox) in enumerate(gt):
            if i in used or gcls != cls:
                continue
            iou = iou_box(box, gbox)
            if iou >= best_iou:
                best_iou = iou
                best_i = i
        if best_i is not None:
            used.add(best_i)
            tp += 1
    fp = len(pred) - tp
    fn = len(gt) - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {"precision": prec, "recall": rec, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


def dice_iou_polygons(a: list[list[float]], b: list[list[float]]) -> dict[str, float]:
    """Coarse proxy using bounding boxes of polygons (not pixel Dice)."""
    def bb(poly: list[list[float]]) -> list[float]:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0, y0 = min(xs), min(ys)
        return [x0, y0, max(xs) - x0, max(ys) - y0]

    if not a or not b:
        return {"iou": 0.0, "dice": 0.0, "note": "polygon_bbox_proxy"}
    iou = iou_box(bb(a), bb(b))
    dice = 2 * iou / (1 + iou) if iou > 0 else 0.0
    return {"iou": iou, "dice": dice, "note": "polygon_bbox_proxy_not_pixel"}


def mae_rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    pairs = [(float(a), float(b)) for a, b in zip(y_true, y_pred, strict=False)]
    if not pairs:
        return {"mae": 0.0, "rmse": 0.0, "n": 0.0}
    err = [p - t for t, p in pairs]
    mae = sum(abs(e) for e in err) / len(err)
    rmse = (sum(e * e for e in err) / len(err)) ** 0.5
    return {"mae": mae, "rmse": rmse, "n": float(len(err))}


def class_counts(labels: Sequence[str]) -> dict[str, int]:
    return dict(Counter(labels))
