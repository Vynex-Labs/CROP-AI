"""Image loading and quality gates. Fail closed, never crash the caller."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

from cropai.config.loader import load_crop_config
from cropai.vision.schema import QualityReport


def load_image(path: str | Path) -> tuple[Image.Image | None, QualityReport]:
    path = Path(path)
    if not path.exists():
        return None, QualityReport(ok=False, score=0.0, reasons=["missing_file"], action="reject")
    try:
        img = Image.open(path)
        img.load()
        img = img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, QualityReport(
            ok=False,
            score=0.0,
            reasons=[f"corrupted:{exc}"],
            action="reject",
        )
    return img, assess_quality(img)


def assess_quality(img: Image.Image) -> QualityReport:
    cfg = load_crop_config()
    reject_floor = float((cfg.get("confidence_thresholds") or {}).get("image_quality_reject", 0.30))
    width, height = img.size
    reasons: list[str] = []
    score = 1.0
    if width < 64 or height < 64:
        reasons.append("too_small")
        score = min(score, 0.1)
    if width > 8192 or height > 8192:
        reasons.append("too_large")
        score = min(score, 0.4)
    gray = img.convert("L")
    stats = ImageStat.Stat(gray)
    mean = float(stats.mean[0]) if stats.mean else 0.0
    stdev = float(stats.stddev[0]) if stats.stddev else 0.0
    if mean < 18:
        reasons.append("too_dark")
        score = min(score, 0.25)
    if mean > 245:
        reasons.append("too_bright")
        score = min(score, 0.25)
    if stdev < 8:
        reasons.append("low_contrast")
        score = min(score, 0.35)
    # Cheap blur proxy: variance of Laplacian-like residual.
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_std = float(ImageStat.Stat(edges).stddev[0] or 0.0)
    if edge_std < 4:
        reasons.append("blurred")
        score = min(score, 0.35)
    action = "ok"
    ok = True
    if score < reject_floor or "too_small" in reasons or any(r.startswith("corrupted") for r in reasons):
        ok = False
        action = "reject" if ("too_small" in reasons or width == 0) else "request_better_image"
        if score < reject_floor and action == "ok":
            action = "request_better_image"
            ok = False
    if reasons and score < 0.5 and action == "ok":
        action = "request_better_image"
        ok = False
    return QualityReport(
        ok=ok,
        score=round(score, 3),
        width=width,
        height=height,
        reasons=reasons,
        action=action,
    )
