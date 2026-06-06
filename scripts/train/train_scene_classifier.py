"""Train the skatepark scene classifier from a manifest.

Usage:
    python scripts/train/train_scene_classifier.py \
        --manifest data/scene/manifest.json

Saves a state_dict to notebooks/models/scene_classifier.pth. Enable it at
runtime by setting SCENE_GUARD_MODEL to that path in .env. CPU is fine — the
model is tiny. See experiments/scene_classifier_journal.md.
"""
import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from trueskate_ai.vision.scene_classifier import SceneClassifier, preprocess  # noqa: E402


class SceneDataset(Dataset):
    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        x = preprocess(it["path"]).squeeze(0)  # (1,H,W)
        y = torch.tensor([float(it["label"])])
        return x, y


def _run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += ((torch.sigmoid(logits) >= 0.5) == (y >= 0.5)).sum().item()
        n += x.size(0)
    torch.set_grad_enabled(True)
    return total_loss / max(1, n), correct / max(1, n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the scene classifier.")
    parser.add_argument("--manifest", type=Path, default=_REPO_ROOT / "data" / "scene" / "manifest.json")
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "notebooks" / "models" / "scene_classifier.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = json.loads(args.manifest.read_text())
    train_items, val_items = data["train"], data["val"]
    print(f"Train {len(train_items)} | Val {len(val_items)}")

    train_loader = DataLoader(SceneDataset(train_items), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(SceneDataset(val_items), batch_size=args.batch_size)

    model = SceneClassifier().to(args.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, criterion, optimizer, args.device, True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, args.device, False)
        print(f"epoch {epoch:02d}  train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}")
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out)

    print(f"\nBest val acc {best_val_acc:.3f}. Saved best model to {args.out}")
    print(f"Enable it: add SCENE_GUARD_MODEL={args.out} to .env")


if __name__ == "__main__":
    main()
