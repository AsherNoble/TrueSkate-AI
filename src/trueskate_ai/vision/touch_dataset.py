import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


@dataclass
class TouchAnnotation:
    filename: str
    x1: Optional[float]
    y1: Optional[float]
    x2: Optional[float]
    y2: Optional[float]


def load_annotations(csv_path: Path) -> List[TouchAnnotation]:
    annotations = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            if x1 < 0 or y1 < 0:
                annotations.append(TouchAnnotation(row["filename"], None, None, None, None))
            elif x2 < 0 or y2 < 0:
                annotations.append(TouchAnnotation(row["filename"], x1, y1, x2, y2))
            else:
                annotations.append(TouchAnnotation(row["filename"], x1, y1, None, None))
    return annotations


def make_heatmap(
    x1: Optional[float],
    y1: Optional[float],
    x2: Optional[float],
    y2: Optional[float],
    H: int,
    W: int,
    sigma: float = 5.0,
) -> np.ndarray:
    """
    Returns a (H, W) float32 heatmap with a Gaussian centred at (x, y).
    If x or y is None, returns an all-zero heatmap.
    """
    heatmap = np.zeros((H, W), dtype=np.float32)

    if x1 is None or y1 is None:
        return heatmap

    # Create coordinate grid
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    # Gaussian bump
    # (x, y) is assumed to be in pixel coords in [0, W-1], [0, H-1]
    dist_sq = (xx - x) ** 2 + (yy - y) ** 2
    heatmap = np.exp(-dist_sq / (2 * sigma ** 2))

    # Normalise to [0, 1] (peak ~1)
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    return heatmap


class TouchDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_csv: str,
        img_size: Tuple[int, int] = (256, 256),
    ):
        """
        images_dir: directory containing frames
        annotations_csv: CSV with columns: filename, x, y (x,y=-1 for no touch)
        img_size: (H, W) the size you want to resize images/heatmaps to
        """
        self.images_dir = Path(images_dir)
        self.annotations = load_annotations(Path(annotations_csv))
        self.img_size = img_size

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = self.images_dir / ann.filename

        # Load image (RGB)
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize to desired size
        H, W = self.img_size
        img = img.resize((W, H), resample=Image.BILINEAR)

        # Scale point coords to new resolution
        if ann.x is not None and ann.y is not None:
            scale_x = W / orig_w
            scale_y = H / orig_h
            x_resized = ann.x * scale_x
            y_resized = ann.y * scale_y
        else:
            x_resized = None
            y_resized = None

        # Create heatmap
        heatmap = make_heatmap(x_resized, y_resized, H, W, sigma=5.0)

        # Convert to tensors
        img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3], range [0,1]
        img_np = np.transpose(img_np, (2, 0, 1))          # [3, H, W]

        img_tensor = torch.from_numpy(img_np)             # float32
        heatmap_tensor = torch.from_numpy(heatmap)[None]  # [1, H, W]

        sample = {
            "image": img_tensor,
            "heatmap": heatmap_tensor,
            "filename": ann.filename,
        }

        return sample