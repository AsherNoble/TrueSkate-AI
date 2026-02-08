"""
Gaussian Bump Predictor - PyTorch Model

This script trains a CNN model to predict gaussian bumps on images.
Input: RGB images
Output: Heatmap with up to 2 gaussian bumps

Usage:
    Training: python gaussian_bump_predictor.py --mode train --data_dir ./data --epochs 100
    Inference: python gaussian_bump_predictor.py --mode infer --model_path model.pth --image_path test.jpg
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================================
# Gaussian Utilities
# ============================================================================

def create_gaussian_heatmap(
    height: int,
    width: int,
    centers: List[Tuple[float, float]],
    sigma: float = 10.0
) -> np.ndarray:
    """
    Create a heatmap with gaussian bumps at specified centers.

    Args:
        height: Height of the heatmap
        width: Width of the heatmap
        centers: List of (x, y) coordinates (normalized 0-1) for bump centers
        sigma: Standard deviation of gaussian bumps

    Returns:
        Heatmap array of shape (height, width) with values in [0, 1]
    """
    heatmap = np.zeros((height, width), dtype=np.float32)

    if not centers:
        return heatmap

    y_grid, x_grid = np.ogrid[0:height, 0:width]

    for cx_norm, cy_norm in centers:
        cx = cx_norm * width
        cy = cy_norm * height

        gaussian = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2))
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap


def extract_bump_centers(
    heatmap: np.ndarray,
    threshold: float = 0.5,
    max_bumps: int = 2
) -> List[Tuple[float, float]]:
    """
    Extract gaussian bump centers from a predicted heatmap.

    Args:
        heatmap: Predicted heatmap of shape (H, W)
        threshold: Minimum value to consider as a bump
        max_bumps: Maximum number of bumps to extract

    Returns:
        List of (x, y) normalized coordinates
    """
    from scipy import ndimage

    height, width = heatmap.shape

    # Find local maxima
    data_max = ndimage.maximum_filter(heatmap, size=5)
    maxima = (heatmap == data_max) & (heatmap > threshold)

    # Get coordinates of maxima
    coords = np.where(maxima)
    if len(coords[0]) == 0:
        return []

    # Get values at maxima and sort by intensity
    values = heatmap[coords]
    sorted_indices = np.argsort(values)[::-1][:max_bumps]

    centers = []
    for idx in sorted_indices:
        y, x = coords[0][idx], coords[1][idx]
        centers.append((x / width, y / height))

    return centers


# ============================================================================
# Dataset
# ============================================================================

class GaussianBumpDataset(Dataset):
    """
    Dataset for gaussian bump prediction.

    Expects a directory structure:
        data_dir/
            images/
                image_001.jpg
                image_002.jpg
                ...
            annotations.json

    annotations.json format:
    {
        "image_001.jpg": [[0.3, 0.4], [0.7, 0.6]],  # Two bumps at normalized coords
        "image_002.jpg": [[0.5, 0.5]],               # One bump
        "image_003.jpg": []                          # No bumps
    }
    """

    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        heatmap_size: Tuple[int, int] = (64, 64),
        sigma: float = 3.0,
        transform: Optional[transforms.Compose] = None
    ):
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images"
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

        # Load annotations
        annotations_path = self.data_dir / "annotations.json"
        if annotations_path.exists():
            with open(annotations_path, 'r') as f:
                self.annotations = json.load(f)
        else:
            # If no annotations, create empty ones for all images
            self.annotations = {}
            if self.image_dir.exists():
                for img_path in self.image_dir.glob("*.[jp][pn][g]"):
                    self.annotations[img_path.name] = []

        self.image_files = list(self.annotations.keys())

        # Default transform
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.image_files[idx]
        img_path = self.image_dir / img_name

        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        # Create target heatmap
        centers = self.annotations.get(img_name, [])
        centers = [(c[0], c[1]) for c in centers]  # Ensure tuple format

        heatmap = create_gaussian_heatmap(
            self.heatmap_size[0],
            self.heatmap_size[1],
            centers,
            sigma=self.sigma
        )
        heatmap = torch.from_numpy(heatmap).unsqueeze(0)  # Add channel dim

        return image, heatmap


# ============================================================================
# Model Architecture
# ============================================================================

class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class GaussianBumpPredictor(nn.Module):
    """
    U-Net style encoder-decoder for predicting gaussian bump heatmaps.

    Architecture:
        - Encoder: Downsampling path with conv blocks
        - Decoder: Upsampling path with skip connections
        - Output: Single channel heatmap with sigmoid activation
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)
        self.enc4 = ConvBlock(base_channels * 4, base_channels * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 16)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        # Output layer
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output
        out = torch.sigmoid(self.out_conv(d1))

        return out


# ============================================================================
# Loss Functions
# ============================================================================

class GaussianBumpLoss(nn.Module):
    """
    Combined loss for gaussian bump prediction.
    Uses MSE + focal-style weighting for sparse targets.
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Focal-style weighting to handle class imbalance
        pos_mask = target.ge(0.5).float()
        neg_mask = target.lt(0.5).float()

        pos_loss = -pos_mask * ((1 - pred) ** self.alpha) * torch.log(pred + 1e-8)
        neg_loss = -neg_mask * ((1 - target) ** self.beta) * (pred ** self.alpha) * torch.log(1 - pred + 1e-8)

        loss = pos_loss + neg_loss
        return loss.mean()


# ============================================================================
# Training
# ============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    save_path: str = "gaussian_bump_model.pth"
) -> dict:
    """Train the gaussian bump predictor."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    criterion = GaussianBumpLoss()

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Resize output to match target if needed
            if outputs.shape[-2:] != targets.shape[-2:]:
                outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation phase
        val_loss = 0.0
        if val_loader:
            model.eval()
            with torch.no_grad():
                for images, targets in val_loader:
                    images = images.to(device)
                    targets = targets.to(device)

                    outputs = model(images)
                    if outputs.shape[-2:] != targets.shape[-2:]:
                        outputs = F.interpolate(outputs, size=targets.shape[-2:], mode='bilinear', align_corners=False)

                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
        else:
            scheduler.step(train_loss)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, save_path)

        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            msg = f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.4f}"
            if val_loader:
                msg += f" | Val Loss: {val_loss:.4f}"
            print(msg)

    return history


# ============================================================================
# Inference
# ============================================================================

def predict_bumps(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    image_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Predict gaussian bumps for a single image.

    Args:
        model: Trained model
        image_path: Path to input image
        device: Torch device
        image_size: Size to resize image
        threshold: Detection threshold

    Returns:
        Tuple of (heatmap, list of bump centers)
    """
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmap = model(image_tensor)

    heatmap = heatmap.squeeze().cpu().numpy()
    centers = extract_bump_centers(heatmap, threshold=threshold)

    return heatmap, centers


def visualize_prediction(
    image_path: str,
    heatmap: np.ndarray,
    centers: List[Tuple[float, float]],
    save_path: Optional[str] = None
):
    """Visualize the prediction with overlay."""

    image = Image.open(image_path).convert('RGB')
    image = np.array(image)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap, cmap='hot')
    axes[1].set_title('Predicted Heatmap')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    h, w = image.shape[:2]
    heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h)))
    axes[2].imshow(heatmap_resized, cmap='hot', alpha=0.5)

    # Mark centers
    for cx, cy in centers:
        axes[2].scatter(cx * w, cy * h, c='cyan', s=100, marker='x', linewidths=3)

    axes[2].set_title(f'Overlay ({len(centers)} bumps detected)')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close()


def batch_inference(
    model: nn.Module,
    image_dir: str,
    device: torch.device,
    output_path: str = "predictions.json",
    threshold: float = 0.5
) -> dict:
    """Run inference on a directory of images."""

    image_dir = Path(image_dir)
    predictions = {}

    for img_path in image_dir.glob("*.[jp][pn][g]"):
        heatmap, centers = predict_bumps(model, str(img_path), device, threshold=threshold)
        predictions[img_path.name] = [[c[0], c[1]] for c in centers]
        print(f"Processed {img_path.name}: {len(centers)} bumps detected")

    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)

    print(f"Predictions saved to {output_path}")
    return predictions


# ============================================================================
# Demo Data Generation
# ============================================================================

def create_demo_dataset(
    output_dir: str,
    num_images: int = 100,
    image_size: int = 256,
    point_radius: int = 3,
    min_points: int = 0,
    max_points: int = 2
):
    """
    Create a synthetic demo dataset with white images and black points.

    Args:
        output_dir: Directory to save the dataset
        num_images: Number of images to generate
        image_size: Size of each image (square)
        point_radius: Radius of black points in pixels
        min_points: Minimum number of points per image
        max_points: Maximum number of points per image (up to 2)
    """

    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    annotations = {}

    for i in range(num_images):
        # Create white image
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

        # Random number of black points (0-2)
        num_points = np.random.randint(min_points, max_points + 1)
        centers = []

        for _ in range(num_points):
            # Random position (keep away from edges)
            margin = 0.05
            cx = np.random.uniform(margin, 1.0 - margin)
            cy = np.random.uniform(margin, 1.0 - margin)
            centers.append([cx, cy])

            # Draw black point
            px, py = int(cx * image_size), int(cy * image_size)

            y, x = np.ogrid[:image_size, :image_size]
            mask = ((x - px) ** 2 + (y - py) ** 2) <= point_radius ** 2
            img[mask] = 0  # Black

        # Save image
        img_name = f"image_{i:04d}.png"
        Image.fromarray(img).save(images_dir / img_name)
        annotations[img_name] = centers

    # Save annotations
    with open(output_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)

    # Print statistics
    num_with_0 = sum(1 for c in annotations.values() if len(c) == 0)
    num_with_1 = sum(1 for c in annotations.values() if len(c) == 1)
    num_with_2 = sum(1 for c in annotations.values() if len(c) == 2)

    print(f"Created demo dataset in {output_dir}")
    print(f"  Total images: {num_images}")
    print(f"  Images with 0 points: {num_with_0}")
    print(f"  Images with 1 point:  {num_with_1}")
    print(f"  Images with 2 points: {num_with_2}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Gaussian Bump Predictor")
    parser.add_argument("--mode", type=str, choices=["train", "infer", "demo"], default="train",
                        help="Mode: train, infer, or demo (create synthetic data)")
    parser.add_argument("--data_dir", type=str, default="./data/gaussian_bumps",
                        help="Directory containing training data")
    parser.add_argument("--model_path", type=str, default="gaussian_bump_model.pth",
                        help="Path to save/load model")
    parser.add_argument("--image_path", type=str, help="Image path for inference")
    parser.add_argument("--image_dir", type=str, help="Directory of images for batch inference")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--image_size", type=int, default=256, help="Input image size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--num_demo_images", type=int, default=100, help="Number of demo images")
    parser.add_argument("--point_radius", type=int, default=3, help="Radius of black points in demo images")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.mode == "demo":
        create_demo_dataset(
            output_dir=args.data_dir,
            num_images=args.num_demo_images,
            image_size=args.image_size,
            point_radius=args.point_radius
        )
        return

    if args.mode == "train":
        # Create dataset
        dataset = GaussianBumpDataset(
            args.data_dir,
            image_size=(args.image_size, args.image_size)
        )

        # Split into train/val
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # Create and train model
        model = GaussianBumpPredictor()
        history = train_model(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, save_path=args.model_path
        )

        # Plot training history
        plt.figure(figsize=(10, 5))
        plt.plot(history["train_loss"], label="Train Loss")
        if history["val_loss"]:
            plt.plot(history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training History")
        plt.savefig("training_history.png")
        print("Training history saved to training_history.png")

    elif args.mode == "infer":
        # Load model
        model = GaussianBumpPredictor()
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        if args.image_path:
            # Single image inference
            heatmap, centers = predict_bumps(
                model, args.image_path, device,
                image_size=(args.image_size, args.image_size),
                threshold=args.threshold
            )
            print(f"Detected {len(centers)} bump(s): {centers}")
            visualize_prediction(args.image_path, heatmap, centers, "prediction.png")

        elif args.image_dir:
            # Batch inference
            batch_inference(model, args.image_dir, device, threshold=args.threshold)

        else:
            print("Please provide --image_path or --image_dir for inference")


if __name__ == "__main__":
    main()
