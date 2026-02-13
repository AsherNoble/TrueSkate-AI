import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import json

# Load all frames
frame_dir = Path("/mnt/user-data/uploads")
frame_files = sorted(frame_dir.glob("img_*.jpg"))
print(f"Found {len(frame_files)} frames: {[f.name for f in frame_files]}")

frames_bgr = [cv2.imread(str(f)) for f in frame_files]
frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
frames_hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames_bgr]

h, w = frames_gray[0].shape
print(f"Frame size: {w}x{h}")

# ============================================================
# STAGE 1: Raw frame differencing
# ============================================================
print("\n=== STAGE 1: Frame Differencing ===")

diffs_gray = []
for i in range(1, len(frames_gray)):
    diff = cv2.absdiff(frames_gray[i], frames_gray[i-1])
    diffs_gray.append(diff)

# Compute stats on each diff
for i, diff in enumerate(diffs_gray):
    mean_val = diff.mean()
    max_val = diff.max()
    # Count pixels above a threshold
    thresh_count = np.sum(diff > 30)
    print(f"  Diff {i+1:2d} (frame {i+1}->{i+2}): mean={mean_val:.2f}, max={max_val}, pixels>30: {thresh_count}")

# ============================================================
# STAGE 2: Color-based trace isolation (orange/warm hue)
# ============================================================
print("\n=== STAGE 2: Orange Trace Isolation ===")

# The trace is orange/fiery. In HSV:
# Hue: ~5-25 (orange range)
# Saturation: moderate to high
# Value: high (bright)
trace_masks = []
for i, hsv in enumerate(frames_hsv):
    # Orange/fire hue range
    lower_orange = np.array([5, 80, 150])
    upper_orange = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Also catch reddish tones
    lower_red = np.array([0, 80, 150])
    upper_red = np.array([5, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    combined = cv2.bitwise_or(mask, mask_red)
    trace_masks.append(combined)
    
    pixel_count = np.sum(combined > 0)
    if pixel_count > 100:
        # Find centroid
        coords = np.argwhere(combined > 0)  # (row, col)
        cy, cx = coords.mean(axis=0)
        print(f"  Frame {i+1:2d}: {pixel_count:6d} orange pixels, centroid=({cx:.0f}, {cy:.0f})")
    else:
        print(f"  Frame {i+1:2d}: {pixel_count:6d} orange pixels (below threshold)")

# ============================================================
# STAGE 3: Combined - diff + color isolation
# ============================================================
print("\n=== STAGE 3: Diff + Color Combined ===")

# For each consecutive pair, compute color diff in HSV space
# and isolate NEW orange pixels
touch_points = []
for i in range(1, len(frames_hsv)):
    # Frame diff (grayscale)
    diff = diffs_gray[i-1]
    
    # Current frame orange mask
    current_orange = trace_masks[i]
    
    # Previous frame orange mask  
    prev_orange = trace_masks[i-1]
    
    # NEW orange: present in current but NOT in previous (or much brighter)
    new_orange = cv2.bitwise_and(current_orange, cv2.bitwise_not(prev_orange))
    
    # Also: pixels that are both in the diff AND orange in current frame
    diff_thresh = (diff > 20).astype(np.uint8) * 255
    diff_and_orange = cv2.bitwise_and(diff_thresh, current_orange)
    
    # Combine both signals
    combined_signal = cv2.bitwise_or(new_orange, diff_and_orange)
    
    # Clean up with morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_signal = cv2.morphologyEx(combined_signal, cv2.MORPH_OPEN, kernel)
    combined_signal = cv2.morphologyEx(combined_signal, cv2.MORPH_CLOSE, kernel)
    
    pixel_count = np.sum(combined_signal > 0)
    
    if pixel_count > 50:
        coords = np.argwhere(combined_signal > 0)
        cy, cx = coords.mean(axis=0)
        
        # Also find the "leading edge" - the point with highest intensity in the diff
        # within the orange region
        masked_diff = diff.copy()
        masked_diff[combined_signal == 0] = 0
        if masked_diff.max() > 0:
            max_loc = np.unravel_index(masked_diff.argmax(), masked_diff.shape)
            lead_y, lead_x = max_loc
        else:
            lead_x, lead_y = cx, cy
        
        touch_points.append({
            'frame': i + 1,
            'is_touching': True,
            'centroid': (float(cx), float(cy)),
            'leading_edge': (float(lead_x), float(lead_y)),
            'pixel_count': int(pixel_count),
            'confidence': min(1.0, pixel_count / 500)
        })
        print(f"  Frame {i+1:2d}: TOUCH centroid=({cx:.0f},{cy:.0f}) leading=({lead_x:.0f},{lead_y:.0f}) pixels={pixel_count} conf={touch_points[-1]['confidence']:.2f}")
    else:
        touch_points.append({
            'frame': i + 1,
            'is_touching': False,
            'centroid': None,
            'leading_edge': None,
            'pixel_count': int(pixel_count),
            'confidence': 0.0
        })
        print(f"  Frame {i+1:2d}: NO TOUCH (pixels={pixel_count})")

# ============================================================
# STAGE 4: Alternative - brightness peak within trace
# ============================================================
print("\n=== STAGE 4: Brightness Peak in Trace Region ===")

for i, (hsv, mask) in enumerate(zip(frames_hsv, trace_masks)):
    if np.sum(mask > 0) > 100:
        # Get value channel (brightness) within orange region
        v_channel = hsv[:, :, 2].copy()
        v_channel[mask == 0] = 0
        
        # Find brightest point
        max_loc = np.unravel_index(v_channel.argmax(), v_channel.shape)
        by, bx = max_loc
        brightness = v_channel[by, bx]
        print(f"  Frame {i+1:2d}: Brightest orange at ({bx},{by}), V={brightness}")

# ============================================================
# VISUALIZATION
# ============================================================
print("\n=== Generating Visualizations ===")

# --- VIZ 1: Frame diff heatmaps (4x5 grid) ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Raw Frame Differences (|frame[N] - frame[N-1]|)', fontsize=16, fontweight='bold')
for i, diff in enumerate(diffs_gray):
    ax = axes[i // 5][i % 5]
    ax.imshow(diff, cmap='hot', vmin=0, vmax=100)
    ax.set_title(f'Diff {i+1}→{i+2}\nmax={diff.max()}, mean={diff.mean():.1f}', fontsize=9)
    ax.axis('off')
# Last cell empty since we have 19 diffs
axes[3][4].axis('off')
plt.tight_layout()
plt.savefig('/home/claude/viz1_frame_diffs.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 2: Orange trace masks (4x5 grid) ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Orange/Fire Trace Isolation (HSV Color Filtering)', fontsize=16, fontweight='bold')
for i, mask in enumerate(trace_masks):
    ax = axes[i // 5][i % 5]
    # Overlay mask on frame
    overlay = frames_rgb[i].copy()
    overlay[mask > 0] = [255, 100, 0]  # highlight orange pixels
    ax.imshow(overlay)
    pixel_count = np.sum(mask > 0)
    ax.set_title(f'Frame {i+1}: {pixel_count} px', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('/home/claude/viz2_trace_masks.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 3: Touch point extraction with combined method ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Extracted Touch Points (Diff + Color Combined)', fontsize=16, fontweight='bold')
for i, tp in enumerate(touch_points):
    ax = axes[i // 5][i % 5]
    frame_img = frames_rgb[i + 1].copy()  # tp corresponds to frame i+2 (index i+1)
    ax.imshow(frame_img)
    
    if tp['is_touching']:
        cx, cy = tp['centroid']
        lx, ly = tp['leading_edge']
        ax.plot(cx, cy, 'go', markersize=12, markeredgewidth=2, markeredgecolor='white', label='Centroid')
        ax.plot(lx, ly, 'r*', markersize=15, markeredgewidth=1, markeredgecolor='white', label='Leading edge')
        ax.set_title(f'Frame {tp["frame"]}: TOUCH\nconf={tp["confidence"]:.2f}', fontsize=9, color='green')
    else:
        ax.set_title(f'Frame {tp["frame"]}: no touch', fontsize=9, color='red')
    ax.axis('off')
# Fill remaining cells
for j in range(len(touch_points), 20):
    axes[j // 5][j % 5].axis('off')
plt.tight_layout()
plt.savefig('/home/claude/viz3_touch_points.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 4: Zoomed trace analysis for key frames ---
# Focus on frames where the trace is most visible (frames 11-20)
interesting_frames = list(range(10, min(20, len(frames_rgb))))  # 0-indexed
fig, axes = plt.subplots(2, len(interesting_frames), figsize=(30, 10))
fig.suptitle('Zoomed Trace Analysis: Key Frames (Top: Original, Bottom: Diff + Trace Highlight)', fontsize=14, fontweight='bold')

for col, fi in enumerate(interesting_frames):
    # Top row: original frame cropped to board area
    ax_top = axes[0][col] if len(interesting_frames) > 1 else axes[0]
    frame_crop = frames_rgb[fi][int(h*0.3):int(h*0.8), :]  # crop to middle
    ax_top.imshow(frame_crop)
    ax_top.set_title(f'Frame {fi+1}', fontsize=9)
    ax_top.axis('off')
    
    # Bottom row: diff with trace overlay
    ax_bot = axes[1][col] if len(interesting_frames) > 1 else axes[1]
    if fi > 0:
        diff = diffs_gray[fi - 1]
        diff_crop = diff[int(h*0.3):int(h*0.8), :]
        
        # Create RGB diff visualization
        diff_vis = np.zeros((*diff_crop.shape, 3), dtype=np.uint8)
        diff_vis[:, :, 0] = np.clip(diff_crop * 3, 0, 255)  # red channel
        diff_vis[:, :, 1] = np.clip(diff_crop * 1, 0, 255)  # green (dimmer)
        
        # Overlay orange mask
        mask_crop = trace_masks[fi][int(h*0.3):int(h*0.8), :]
        diff_vis[mask_crop > 0, 1] = 200  # green tint on orange pixels
        
        ax_bot.imshow(diff_vis)
    ax_bot.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/viz4_zoomed_trace.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 5: Summary trajectory plot ---
fig, ax = plt.subplots(1, 1, figsize=(10, 16))
# Show last frame as background
ax.imshow(frames_rgb[-1], alpha=0.5)

touching = [tp for tp in touch_points if tp['is_touching']]
if touching:
    centroids_x = [tp['centroid'][0] for tp in touching]
    centroids_y = [tp['centroid'][1] for tp in touching]
    leads_x = [tp['leading_edge'][0] for tp in touching]
    leads_y = [tp['leading_edge'][1] for tp in touching]
    frames_touching = [tp['frame'] for tp in touching]
    
    # Color by frame number
    colors = plt.cm.plasma(np.linspace(0, 1, len(touching)))
    
    ax.scatter(centroids_x, centroids_y, c=colors, s=100, marker='o', edgecolors='white', linewidth=1.5, zorder=5, label='Centroid')
    ax.scatter(leads_x, leads_y, c=colors, s=150, marker='*', edgecolors='white', linewidth=1, zorder=6, label='Leading edge')
    
    # Connect with line
    ax.plot(centroids_x, centroids_y, 'g--', alpha=0.7, linewidth=2, zorder=4)
    
    # Label frames
    for tp, color in zip(touching, colors):
        cx, cy = tp['centroid']
        ax.annotate(f"F{tp['frame']}", (cx, cy), textcoords="offset points", 
                    xytext=(10, -10), fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

ax.set_title('Extracted Touch Trajectory Over Time', fontsize=14, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig('/home/claude/viz5_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved!")
print(f"\nSummary: {len(touching)} frames with detected touch out of {len(touch_points)} frame pairs analyzed")

# Save touch data as JSON
output_data = {
    'total_frames': len(frames_rgb),
    'frame_pairs_analyzed': len(touch_points),
    'frames_with_touch': len(touching),
    'touch_points': touch_points
}
with open('/home/claude/touch_data.json', 'w') as f:
    json.dump(output_data, f, indent=2)
print("Touch data saved to touch_data.json")
