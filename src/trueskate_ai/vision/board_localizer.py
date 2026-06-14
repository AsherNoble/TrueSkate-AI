"""Classical-CV skateboard localizer for True Skate frames.

The leap from single tricks to SEQUENCES needs board-relative gestures: to fire
trick B after trick A, you must aim B at wherever the board ended up. That
requires perceiving the board's pose (centre, heading, size) from a frame. This
module is the first, debuggable, training-free version of that perception — a
keystone for Option D (board localizer + 2-trick lines) and a baseline the
learned models can later beat.

The True Skate deck is a dark, elongated grip-taped object sitting on a bright,
low-texture floor, near the lower-centre of a portrait frame. We isolate the
dark region inside a region-of-interest (excluding the top UI bar), take the
largest sufficiently-elongated blob near the horizontal centre, and fit a
min-area rectangle → centre + heading + length.

Coordinates returned are normalised [0, 1] (see GESTURES.md), so a pose maps
directly onto the gesture coordinate system used by scale_to_device().
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardPose:
    """Localized board pose in normalised [0, 1] frame coordinates.

    cx, cy:      board centroid (normalised).
    angle_deg:   deck long-axis orientation in degrees, 0 = vertical (nose up),
                 positive = clockwise; range (-90, 90].
    length:      deck long-axis length as a fraction of frame height (a proxy
                 for how near/large the board is).
    area_frac:   blob area as a fraction of the full frame.
    confidence:  heuristic [0, 1] — elongation × centrality × size plausibility.
    """
    cx: float
    cy: float
    angle_deg: float
    length: float
    area_frac: float
    confidence: float


@dataclass(frozen=True)
class BoardLocatorConfig:
    # Defaults tuned for LIVE in-park frames (SLS Super Crown): the tighter band
    # excludes the bottom menu bar and the top scoreboard/ledge that otherwise
    # win the dark-blob contest. The surround/saturation scoring (below) does the
    # rest — it rejects dark UI strips and obstacle structures by preferring the
    # deck (dark, on bright floor, with saturated wheels/holographic edges).
    roi_top: float = 0.28        # ignore the top UI/scoreboard/ledge band
    roi_bottom: float = 0.88     # ignore the bottom controls/menu strip
    roi_left: float = 0.06
    roi_right: float = 0.94
    min_area_frac: float = 0.004  # board must occupy at least this fraction of the frame
    max_area_frac: float = 0.22   # ...and at most this (reject whole-frame dark)
    min_elongation: float = 1.6   # long/short side ratio of the min-area rect
    blur_ksize: int = 5
    close_ksize: int = 9


def _largest_components(mask: np.ndarray, top_k: int = 6):
    """Return up to top_k (label, area) of the largest foreground components."""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
    comps.sort(key=lambda t: -t[1])
    return comps[:top_k], labels, stats, centroids


def locate_board(bgr: np.ndarray, config: BoardLocatorConfig | None = None) -> BoardPose | None:
    """Locate the skateboard deck in a True Skate BGR frame.

    Returns a BoardPose in normalised coordinates, or None if no plausible
    board blob is found. Pure heuristic — no learned weights.
    """
    cfg = config or BoardLocatorConfig()
    h, w = bgr.shape[:2]
    frame_area = float(h * w)

    # --- region of interest (exclude UI bands / edges) ---
    y0, y1 = int(cfg.roi_top * h), int(cfg.roi_bottom * h)
    x0, x1 = int(cfg.roi_left * w), int(cfg.roi_right * w)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)

    # The floor is bright; the deck is darker. Otsu splits them; we keep the
    # dark side. THRESH_BINARY_INV → deck (dark) becomes foreground (255).
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_ksize, cfg.close_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (27, 27))

    comps, labels, stats, _ = _largest_components(mask)
    if not comps:
        return None

    best = None
    best_score = 0.0
    roi_h, roi_w = mask.shape
    for label, area in comps:
        area_frac = area / frame_area
        if area_frac < cfg.min_area_frac or area_frac > cfg.max_area_frac:
            continue
        comp_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        if len(contour) < 5:
            continue
        (rcx, rcy), (rw, rh), angle = cv2.minAreaRect(contour)
        long_side, short_side = max(rw, rh), max(1.0, min(rw, rh))
        elongation = long_side / short_side
        if elongation < cfg.min_elongation:
            continue

        # Full-frame normalised centroid.
        cx = (x0 + rcx) / w
        cy = (y0 + rcy) / h
        # Centrality: prefer blobs near the horizontal centre (board sits there).
        centrality = 1.0 - min(1.0, abs(cx - 0.5) / 0.5)
        # Size plausibility: peak around a mid fraction, falls off at extremes.
        size_score = float(np.clip(area_frac / 0.05, 0.0, 1.0))
        # Bright surround: the deck sits on bright flat floor; a ring around it is
        # bright, unlike dark UI strips (menu bar) whose surround is dark.
        dilated = cv2.dilate(comp_mask, ring_kernel)
        ring = (dilated > 0) & (comp_mask == 0)
        surround = float(gray[ring].mean() / 255.0) if ring.any() else 0.0
        # Saturated colour near the blob: the deck has saturated wheels +
        # holographic grip edges; plain dark structures do not.
        bx, by = stats[label, cv2.CC_STAT_LEFT], stats[label, cv2.CC_STAT_TOP]
        bw, bh = stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]
        m = 18
        patch = hsv[max(0, by - m):by + bh + m, max(0, bx - m):bx + bw + m]
        sat = float(((patch[:, :, 1] > 90) & (patch[:, :, 2] > 90)).mean()) if patch.size else 0.0
        score = (
            (min(elongation, 6.0) / 6.0)
            * (0.3 + 0.7 * centrality)
            * (0.3 + 0.7 * size_score)
            * (0.25 + 0.75 * surround)
            * (0.4 + 0.6 * min(1.0, sat * 6.0))
        )

        if score > best_score:
            # cv2 minAreaRect angle convention → normalise to 0 = vertical long axis.
            if rw < rh:
                norm_angle = angle
            else:
                norm_angle = angle - 90.0
            # wrap to (-90, 90]
            while norm_angle <= -90.0:
                norm_angle += 180.0
            while norm_angle > 90.0:
                norm_angle -= 180.0
            best_score = score
            best = BoardPose(
                cx=round(cx, 4),
                cy=round(cy, 4),
                angle_deg=round(float(norm_angle), 2),
                length=round(long_side / h, 4),
                area_frac=round(area_frac, 4),
                confidence=round(float(min(1.0, score)), 3),
            )

    return best


def reanchor_gesture_points(
    points: list[tuple[float, float]],
    ref_pose: BoardPose,
    cur_pose: BoardPose,
    *,
    rotate: bool = True,
) -> list[tuple[float, float]]:
    """Re-aim a gesture authored for ``ref_pose`` onto the board's ``cur_pose``.

    This is the core primitive for trick SEQUENCES: a converged recipe is
    authored relative to where the board sat when it was learned (ref_pose);
    to fire it as the next trick in a line, transform its (normalised) waypoints
    by the rigid motion that takes ref_pose → cur_pose (translate the centroid,
    optionally rotate by the heading delta about the centroid).

    Pure function — no device, fully unit-testable. Points and poses are in
    normalised [0, 1] frame coordinates. Rotation is applied in normalised
    space (a mild shear vs. true pixels, negligible for small heading deltas;
    refine with the device aspect ratio if needed).
    """
    dcx = cur_pose.cx - ref_pose.cx
    dcy = cur_pose.cy - ref_pose.cy
    if rotate:
        dtheta = np.deg2rad(cur_pose.angle_deg - ref_pose.angle_deg)
        cos_t, sin_t = float(np.cos(dtheta)), float(np.sin(dtheta))
    out: list[tuple[float, float]] = []
    for x, y in points:
        # translate the whole gesture by the centroid delta
        nx, ny = x + dcx, y + dcy
        if rotate:
            # rotate about the (new) board centroid
            ox, oy = nx - cur_pose.cx, ny - cur_pose.cy
            rx = ox * cos_t - oy * sin_t
            ry = ox * sin_t + oy * cos_t
            nx, ny = cur_pose.cx + rx, cur_pose.cy + ry
        out.append((float(nx), float(ny)))
    return out


def draw_pose(bgr: np.ndarray, pose: BoardPose | None) -> np.ndarray:
    """Return a copy of bgr with the board pose overlaid (for QA)."""
    out = bgr.copy()
    h, w = out.shape[:2]
    if pose is None:
        cv2.putText(out, "no board", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return out
    cx, cy = int(pose.cx * w), int(pose.cy * h)
    half = int(pose.length * h / 2)
    theta = np.deg2rad(pose.angle_deg)
    # long axis: 0deg = vertical, so direction = (sin, -cos)... use (sin, cos) downward
    dx, dy = np.sin(theta), -np.cos(theta)
    p1 = (int(cx - half * dx), int(cy - half * dy))
    p2 = (int(cx + half * dx), int(cy + half * dy))
    cv2.line(out, p1, p2, (0, 255, 0), 4)
    cv2.circle(out, (cx, cy), 12, (0, 165, 255), -1)
    cv2.putText(out, f"({pose.cx:.2f},{pose.cy:.2f}) {pose.angle_deg:+.0f}deg conf={pose.confidence:.2f}",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return out


# --- CLI: validate on a directory of frames -------------------------------
if __name__ == "__main__":
    import argparse
    import glob
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Validate the board localizer on a frame directory.")
    ap.add_argument("frames_glob", help="Glob of frame images, e.g. 'data/extracted_frames/laser_flip_001_60fps/img_*.jpg'")
    ap.add_argument("--out-dir", default="tmp/board_localizer", help="Where to write annotated overlays")
    ap.add_argument("--save-every", type=int, default=10, help="Save an overlay every Nth frame")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.frames_glob))
    if not paths:
        raise SystemExit(f"no frames match {args.frames_glob}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    found = 0
    confs = []
    for i, p in enumerate(paths):
        img = cv2.imread(p)
        pose = locate_board(img)
        if pose is not None:
            found += 1
            confs.append(pose.confidence)
        if i % args.save_every == 0:
            cv2.imwrite(str(out_dir / f"overlay_{i:04d}.png"), draw_pose(img, pose))
    rate = 100 * found / len(paths)
    mean_conf = float(np.mean(confs)) if confs else 0.0
    print(f"{Path(args.frames_glob).parent.name}: {found}/{len(paths)} frames localized ({rate:.0f}%), "
          f"mean conf={mean_conf:.2f}; overlays → {out_dir}")
