"""Shared adapter helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from cropai.config.loader import load_yaml
from cropai.dataset.schema import ImageRecord
from cropai.utils.paths import repo_root


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class SourceAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def ingest(self, root: Path, **kwargs: Any) -> list[ImageRecord]:
        raise NotImplementedError

    def available(self, root: Path) -> bool:
        return root.exists() and any(root.rglob("*"))


def load_class_map(relative: str) -> dict[str, str]:
    path = repo_root() / relative
    if not path.exists():
        return {}
    data = load_yaml(path)
    mapping = dict(data.get("map") or {})
    return {str(k): str(v) for k, v in mapping.items()}


def discover_imagefolder(root: Path) -> Iterable[tuple[str, Path]]:
    """Yield (class_folder_name, image_path) for ImageFolder layouts."""
    if not root.exists():
        return
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield class_dir.name, path
