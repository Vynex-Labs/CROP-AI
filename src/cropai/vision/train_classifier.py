"""Classifier training. Transfer learning; never claimed complete without user GPU run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cropai.config.loader import load_dataset_config, load_training_config
from cropai.dataset.schema import read_csv
from cropai.domain.taxonomy import Taxonomy
from cropai.utils.hardware import detect_hardware, format_hardware_report
from cropai.utils.logging import setup_logging
from cropai.utils.paths import data_dir, repo_root, runs_dir
from cropai.vision.backends import torch_available
from cropai.vision.metrics import classification_report
from cropai.vision.run_meta import collect_run_meta, write_run_meta


log = setup_logging()


def _split_csv(split: str, version: str | None = None) -> Path:
    cfg = load_dataset_config()
    version = version or str(cfg.get("version") or "v0.1.1-phase1")
    return data_dir() / "splits" / version / f"{split}.csv"


def plan(arch: str = "efficientnet_v2_s") -> dict[str, Any]:
    tcfg = load_training_config()
    hw = detect_hardware()
    tax = Taxonomy()
    return {
        "task": "classification",
        "candidate": arch or tcfg["classifier"]["candidate"],
        "alternatives": tcfg["classifier"]["alternatives"],
        "n_classes": len(tax.classifier_classes()),
        "image_size": tcfg["classifier"]["image_size"],
        "epochs": tcfg["classifier"]["epochs"],
        "lr": tcfg["classifier"]["lr"],
        "pretrained": tcfg["classifier"]["pretrained"],
        "torch": torch_available(),
        "cuda": hw.get("cuda_available"),
        "hardware": format_hardware_report(hw),
        "status": "NOT_TRAINED" if not torch_available() else "READY_IF_DATA",
    }


def train_classifier(
    *,
    arch: str | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    device: str = "auto",
    max_samples: int | None = None,
) -> dict[str, Any]:
    tcfg = load_training_config()
    arch = arch or str(tcfg["classifier"]["candidate"])
    summary = plan(arch)
    if dry_run:
        return {**summary, "dry_run": True}
    if not torch_available():
        summary["ok"] = False
        summary["reason"] = "torch_not_installed"
        log.warning("PyTorch not installed. Install requirements-train.txt on the CUDA machine.")
        return summary

    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import functional as F
    from PIL import Image

    from cropai.vision.torch_models import IMAGENET_MEAN, IMAGENET_STD, build_classifier, resolve_device

    device_s = resolve_device(device)
    if device_s != "cuda":
        log.warning("WARNING: CUDA unavailable. Continuing with CPU fallback where practical.")

    tax = Taxonomy()
    classes = tax.classifier_classes()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    image_size = 64 if smoke else int(tcfg["classifier"]["image_size"])
    epochs = 1 if smoke else int(tcfg["classifier"]["epochs"])
    batch = 2 if smoke else int(tcfg["dataloader"]["batch_size"])
    lr = float(tcfg["classifier"]["lr"])
    seed = int(tcfg.get("seed", 42))
    torch.manual_seed(seed)

    class ManifestDataset(Dataset):
        def __init__(self, split: str) -> None:
            path = _split_csv(split)
            self.rows = read_csv(path) if path.exists() else []
            if max_samples:
                self.rows = self.rows[:max_samples]
            if smoke:
                self.rows = self.rows[: max(8, batch * 2)]

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            rec = self.rows[idx]
            img_path = data_dir() / rec.relpath
            if not img_path.exists():
                img_path = repo_root() / rec.relpath
            img = Image.open(img_path).convert("RGB").resize((image_size, image_size))
            x = F.normalize(F.to_tensor(img), IMAGENET_MEAN, IMAGENET_STD)
            y = class_to_idx.get(rec.class_id, 0)
            return x, y

    train_ds = ManifestDataset("train")
    val_ds = ManifestDataset("val")
    if len(train_ds) == 0:
        return {**summary, "ok": False, "reason": "empty_train_split"}

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0) if len(val_ds) else None

    model = build_classifier(arch, len(classes), pretrained=not smoke and bool(tcfg["classifier"]["pretrained"]))
    model.to(device_s)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(tcfg["classifier"]["weight_decay"]))
    loss_fn = torch.nn.CrossEntropyLoss()
    use_amp = bool(tcfg.get("amp", True)) and device_s == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_dir = runs_dir() / "classifier" / f"{arch}_{'smoke' if smoke else 'full'}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    history: list[dict[str, Any]] = []

    def _save(path: Path, extra: dict[str, Any]) -> None:
        torch.save(
            {
                "arch": arch,
                "classes": classes,
                "state_dict": model.state_dict(),
                **extra,
            },
            path,
        )

    freeze_epochs = 0 if smoke else int(tcfg["classifier"].get("freeze_backbone_epochs", 0))

    def _set_freeze(frozen: bool) -> None:
        for name, p in model.named_parameters():
            if "classifier" in name or name.startswith("fc"):
                p.requires_grad = True
            else:
                p.requires_grad = not frozen

    for epoch in range(epochs):
        _set_freeze(epoch < freeze_epochs)
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device_s), yb.to(device_s)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = loss_fn(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * len(xb)
            n += len(xb)
        train_loss = running / max(n, 1)

        y_true: list[str] = []
        y_pred: list[str] = []
        if val_loader:
            model.eval()
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device_s)
                    logits = model(xb)
                    pred = logits.argmax(1).cpu().tolist()
                    y_pred.extend(classes[i] for i in pred)
                    y_true.extend(classes[int(i)] for i in yb.tolist())
        report = classification_report(y_true, y_pred) if y_true else {"macro_f1": 0.0, "accuracy": 0.0}
        row = {"epoch": epoch, "train_loss": train_loss, **{k: report[k] for k in ("accuracy", "macro_f1") if k in report}}
        history.append(row)
        log.info("classifier epoch %s %s", epoch, row)
        _save(last_path, {"epoch": epoch, "metrics": row})
        f1 = float(report.get("macro_f1") or 0.0)
        if f1 >= best_f1:
            if best_path.exists() and best_f1 >= 0:
                preserved = run_dir / f"best_epoch{epoch-1}.pt"
                if not preserved.exists():
                    best_path.replace(preserved)
                    _save(best_path, {"epoch": epoch, "metrics": row})
                else:
                    _save(best_path, {"epoch": epoch, "metrics": row})
            else:
                _save(best_path, {"epoch": epoch, "metrics": row})
            best_f1 = f1

    meta = collect_run_meta(
        dataset_version=load_dataset_config().get("version"),
        dataset_size=len(train_ds),
        model=arch,
        pretrained_checkpoint="IMAGENET_DEFAULT" if tcfg["classifier"]["pretrained"] and not smoke else "none",
        hyperparameters={"lr": lr, "epochs": epochs, "batch": batch, "image_size": image_size, "amp": use_amp},
        seed=seed,
        metrics={"best_macro_f1": best_f1, "history": history},
        checkpoint_path=str(best_path),
        smoke=smoke,
    )
    write_run_meta(run_dir, meta)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    return {**summary, "ok": True, "run_dir": str(run_dir), "best_macro_f1": best_f1, "smoke": smoke}
