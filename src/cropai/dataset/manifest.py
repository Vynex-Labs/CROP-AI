"""Dataset manifests and versioning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cropai.dataset.schema import ImageRecord, read_csv, write_csv
from cropai.utils.hashing import sha256_file, sha256_text
from cropai.utils.logging import utc_now_iso
from cropai.utils.paths import data_dir, repo_root


@dataclass
class Manifest:
    version: str
    created_at: str
    records: list[ImageRecord]
    sources: list[str] = field(default_factory=list)
    notes: str = ""
    git_commit: str = ""
    csv_sha256: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return len(self.records)

    def counts_by(self, attr: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in self.records:
            value = getattr(rec, attr, "")
            if value is None or value == "":
                key = "unknown"
            else:
                key = str(value)
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))

    def synthetic_count(self) -> int:
        return sum(1 for r in self.records if r.is_synthetic)

    def field_count(self) -> int:
        return sum(1 for r in self.records if r.is_field and not r.is_synthetic)

    def real_count(self) -> int:
        return sum(1 for r in self.records if not r.is_synthetic)


def _git_commit() -> str:
    head = repo_root() / ".git" / "HEAD"
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref = repo_root() / ".git" / text.split(" ", 1)[1].strip()
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
        return text
    return text


def write_manifest(manifest: Manifest, csv_path: Path | None = None) -> Path:
    manifests_dir = data_dir() / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_path or (manifests_dir / f"images_{manifest.version}.csv")
    write_csv(csv_path, manifest.records)
    manifest.csv_sha256 = sha256_file(csv_path)
    manifest.git_commit = manifest.git_commit or _git_commit()
    sidecar = {
        "version": manifest.version,
        "created_at": manifest.created_at,
        "n_records": manifest.size,
        "n_synthetic": manifest.synthetic_count(),
        "n_real": manifest.real_count(),
        "n_field_real": manifest.field_count(),
        "sources": manifest.sources,
        "csv": str(csv_path.relative_to(repo_root())) if csv_path.is_relative_to(repo_root()) else str(csv_path),
        "csv_sha256": manifest.csv_sha256,
        "git_commit": manifest.git_commit,
        "counts": {
            "crop_id": manifest.counts_by("crop_id"),
            "class_id": manifest.counts_by("class_id"),
            "source": manifest.counts_by("source"),
            "split": manifest.counts_by("split"),
            "is_field": manifest.counts_by("is_field"),
            "is_synthetic": manifest.counts_by("is_synthetic"),
        },
        "notes": manifest.notes,
        "extra": manifest.extra,
        "fingerprint": sha256_text(manifest.csv_sha256 + manifest.version),
    }
    json_path = manifests_dir / f"images_{manifest.version}.json"
    json_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    latest = manifests_dir / "latest.json"
    latest.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json_path


def read_manifest(csv_path: Path, version: str = "unknown") -> Manifest:
    records = read_csv(csv_path)
    return Manifest(
        version=version,
        created_at=utc_now_iso(),
        records=records,
        sources=sorted({r.source for r in records}),
        csv_sha256=sha256_file(csv_path) if csv_path.exists() else "",
    )


def load_latest_sidecar() -> dict[str, Any] | None:
    path = data_dir() / "manifests" / "latest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
