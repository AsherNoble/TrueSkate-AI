import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

# Load all frames (new batch - 001 to 020)
frame_dir = Path("/mnt/user-data/uploads")
frame_files = sorted(frame_dir.glob("img_*.jpg"))
# Only use the new batch (001-020)
frame_files = [f for f in frame_files if int(f.stem.split('_')[1]) <= 20]
print(f"Found {len(frame_files)} frames: {[f.name for f in frame_files]}")

frames_bgr = [cv2.imread(str(f)) for f in frame_files]
frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
frames_hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames_bgr]

h, w = frames_gray[0].shape
print(f"Frame size: {w}x{h}")

# ============================================================
# STAGE 1: Frame differencing - verify no duplicates
# ============================================================
print("\n=== STAGE 1: Frame Differencing ===")

diffs_gray = []
diffs_color = []  # Keep color diffs too for better analysis
for i in range(1, len(frames_gray)):
    diff = cv2.absdiff(frames_gray[i], frames_gray[i-1])
    diff_color = cv2.absdiff(frames_bgr[i], frames_bgr[i-1])
    diffs_gray.append(diff)
    diffs_color.append(diff_color)
    mean_val = diff.mean()
    max_val = diff.max()
    thresh_count = np.sum(diff > 30)
    print(f"  Diff {i:2d}→{i+1:2d}: mean={mean_val:.2f}, max={max_val}, pixels>30: {thresh_count}")

# ============================================================
# STAGE 2: Orange trace isolation with wider hue range
# ============================================================
print("\n=== STAGE 2: Orange Trace Isolation ===")

trace_masks = []
for i, hsv in enumerate(frames_hsv):
    # Wider orange/fire range to catch the full trace
    lower_orange = np.array([5, 50, 120])
    upper_orange = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    # Reddish tones
    lower_red = np.array([0, 50, 120])
    upper_red = np.array([5, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    
    # Also catch warm yellows/whites that appear at trace center
    lower_yellow = np.array([15, 30, 200])
    upper_yellow = np.array([40, 150, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    combined = cv2.bitwise_or(mask, mask_red)
    combined = cv2.bitwise_or(combined, mask_yellow)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    trace_masks.append(combined)
    
    pixel_count = np.sum(combined > 0)
    if pixel_count > 100:
        coords = np.argwhere(combined > 0)
        cy, cx = coords.mean(axis=0)
        print(f"  Frame {i+1:2d}: {pixel_count:6d} trace pixels, centroid=({cx:.0f}, {cy:.0f})")
    else:
        print(f"  Frame {i+1:2d}: {pixel_count:6d} trace pixels (minimal)")

# ============================================================
# STAGE 3: Combined diff + color - improved for overlapping swipes
# ============================================================
print("\n=== STAGE 3: Diff + Color Combined (improved) ===")

touch_points = []
for i in range(1, len(frames_hsv)):
    diff = diffs_gray[i-1]
    current_orange = trace_masks[i]
    prev_orange = trace_masks[i-1]
    
    # NEW orange pixels (present now, not before)
    new_orange = cv2.bitwise_and(current_orange, cv2.bitwise_not(prev_orange))
    
    # Pixels that changed AND are orange in current frame
    diff_thresh = (diff > 15).astype(np.uint8) * 255
    diff_and_orange = cv2.bitwise_and(diff_thresh, current_orange)
    
    # For overlapping swipes: also look at DISAPPEARED orange
    # (previous swipe fading where new swipe isn't)
    disappeared_orange = cv2.bitwise_and(prev_orange, cv2.bitwise_not(current_orange))
    
    # Combined signal for new touch
    combined_signal = cv2.bitwise_or(new_orange, diff_and_orange)
    
    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_signal = cv2.morphologyEx(combined_signal, cv2.MORPH_OPEN, kernel)
    combined_signal = cv2.morphologyEx(combined_signal, cv2.MORPH_CLOSE, kernel)
    
    pixel_count = np.sum(combined_signal > 0)
    disappeared_count = np.sum(disappeared_orange > 0)
    
    if pixel_count > 50:
        coords = np.argwhere(combined_signal > 0)
        cy, cx = coords.mean(axis=0)
        
        # Find leading edge: brightest point in the diff within orange region
        masked_diff = diff.copy()
        masked_diff[combined_signal == 0] = 0
        if masked_diff.max() > 0:
            max_loc = np.unravel_index(masked_diff.argmax(), masked_diff.shape)
            lead_y, lead_x = max_loc
        else:
            lead_x, lead_y = cx, cy
        
        # Estimate if this is a NEW swipe or continuation
        # A new swipe will have a large new_orange count and possibly disappeared pixels
        new_orange_count = np.sum(new_orange > 0)
        is_new_swipe = new_orange_count > 2000  # Threshold for "suddenly appeared"
        
        touch_points.append({
            'frame': i + 1,
            'is_touching': True,
            'centroid': (float(cx), float(cy)),
            'leading_edge': (float(lead_x), float(lead_y)),
            'pixel_count': int(pixel_count),
            'new_pixels': int(new_orange_count),
            'disappeared_pixels': int(disappeared_count),
            'is_new_swipe': bool(is_new_swipe),
            'confidence': min(1.0, pixel_count / 500)
        })
        marker = " *** NEW SWIPE ***" if is_new_swipe else ""
        print(f"  Frame {i+1:2d}: TOUCH centroid=({cx:.0f},{cy:.0f}) leading=({lead_x:.0f},{lead_y:.0f}) "
              f"new={new_orange_count} disappeared={disappeared_count}{marker}")
    else:
        touch_points.append({
            'frame': i + 1,
            'is_touching': False,
            'centroid': None,
            'leading_edge': None,
            'pixel_count': int(pixel_count),
            'new_pixels': 0,
            'disappeared_pixels': int(disappeared_count),
            'is_new_swipe': False,
            'confidence': 0.0
        })
        print(f"  Frame {i+1:2d}: NO TOUCH (combined={pixel_count}, disappeared={disappeared_count})")

# ============================================================
# STAGE 4: Connected component analysis on trace
# ============================================================
print("\n=== STAGE 4: Connected Components (Multiple Trace Blobs) ===")

for i, mask in enumerate(trace_masks):
    if np.sum(mask > 0) > 200:
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        # Filter out background (label 0) and tiny components
        significant = [(j, stats[j, cv2.CC_STAT_AREA], centroids[j]) 
                       for j in range(1, num_labels) if stats[j, cv2.CC_STAT_AREA] > 100]
        if len(significant) > 1:
            print(f"  Frame {i+1:2d}: {len(significant)} trace blobs:")
            for j, (label, area, centroid) in enumerate(significant):
                print(f"    Blob {j+1}: area={area}, centroid=({centroid[0]:.0f},{centroid[1]:.0f})")
        elif len(significant) == 1:
            _, area, centroid = significant[0]
            print(f"  Frame {i+1:2d}: 1 trace blob, area={area}, centroid=({centroid[0]:.0f},{centroid[1]:.0f})")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n=== Generating Visualizations ===")

# --- VIZ 1: Frame diffs heatmap (4x5) ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Frame Differences at 60fps (Nollie 360 Double Flip)', fontsize=16, fontweight='bold')
for i, diff in enumerate(diffs_gray):
    ax = axes[i // 5][i % 5]
    ax.imshow(diff, cmap='hot', vmin=0, vmax=100)
    ax.set_title(f'Diff {i+1}→{i+2}\nmax={diff.max()}, mean={diff.mean():.1f}', fontsize=9)
    ax.axis('off')
axes[3][4].axis('off')
plt.tight_layout()
plt.savefig('/home/claude/v2_viz1_frame_diffs.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 2: Trace masks with connected components ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Orange Trace Isolation + Connected Components', fontsize=16, fontweight='bold')
for i, mask in enumerate(trace_masks):
    ax = axes[i // 5][i % 5]
    overlay = frames_rgb[i].copy()
    
    # Color different connected components differently
    if np.sum(mask > 0) > 200:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        colors = [(255, 100, 0), (0, 255, 100), (100, 100, 255), (255, 255, 0)]
        for j in range(1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] > 100:
                color = colors[(j-1) % len(colors)]
                overlay[labels == j] = color
                cx, cy = centroids[j]
                ax.plot(cx, cy, 'w+', markersize=10, markeredgewidth=2)
    
    ax.imshow(overlay)
    pixel_count = np.sum(mask > 0)
    ax.set_title(f'Frame {i+1}: {pixel_count} px', fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('/home/claude/v2_viz2_trace_components.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 3: Touch points with swipe detection ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Extracted Touch Points + New Swipe Detection', fontsize=16, fontweight='bold')
for i, tp in enumerate(touch_points):
    ax = axes[i // 5][i % 5]
    frame_img = frames_rgb[i + 1].copy()
    ax.imshow(frame_img)
    
    if tp['is_touching']:
        cx, cy = tp['centroid']
        lx, ly = tp['leading_edge']
        ax.plot(cx, cy, 'go', markersize=12, markeredgewidth=2, markeredgecolor='white')
        ax.plot(lx, ly, 'r*', markersize=15, markeredgewidth=1, markeredgecolor='white')
        
        color = 'magenta' if tp['is_new_swipe'] else 'green'
        label = f'Frame {tp["frame"]}: {"NEW SWIPE" if tp["is_new_swipe"] else "TOUCH"}\n' \
                f'new={tp["new_pixels"]}'
        ax.set_title(label, fontsize=8, color=color)
    else:
        ax.set_title(f'Frame {tp["frame"]}: no touch\ndisappeared={tp["disappeared_pixels"]}', fontsize=8, color='red')
    ax.axis('off')
axes[3][4].axis('off')
plt.tight_layout()
plt.savefig('/home/claude/v2_viz3_touch_swipe.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 4: Key frame comparison (5→6 overlap) ---
fig, axes = plt.subplots(2, 4, figsize=(24, 12))
fig.suptitle('Critical Overlap Analysis: Frames 4→7 (Two Swipes Merge)', fontsize=16, fontweight='bold')

for col, fi in enumerate(range(3, 7)):  # frames 4-7 (0-indexed 3-6)
    # Top: original
    ax_top = axes[0][col]
    ax_top.imshow(frames_rgb[fi])
    ax_top.set_title(f'Frame {fi+1} (Original)', fontsize=10)
    ax_top.axis('off')
    
    # Bottom: diff from previous + trace overlay
    ax_bot = axes[1][col]
    if fi > 0:
        diff = diffs_gray[fi - 1]
        
        # Create composite: diff in red, trace mask in green, new pixels in blue
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        composite[:, :, 0] = np.clip(diff * 2, 0, 255)  # diff in red
        composite[:, :, 1] = trace_masks[fi]  # current trace in green
        if fi > 1:
            # New orange only
            new_only = cv2.bitwise_and(trace_masks[fi], cv2.bitwise_not(trace_masks[fi-1]))
            composite[:, :, 2] = new_only  # new trace pixels in blue
        
        ax_bot.imshow(composite)
        ax_bot.set_title(f'Red=diff, Green=trace, Blue=NEW trace', fontsize=9)
    ax_bot.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/v2_viz4_overlap_detail.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 5: Full trajectory ---
fig, ax = plt.subplots(1, 1, figsize=(10, 16))
# Show a mid-sequence frame as background
mid_frame = len(frames_rgb) // 3  # ~frame 6-7 where trace is most visible
ax.imshow(frames_rgb[mid_frame], alpha=0.5)

touching = [tp for tp in touch_points if tp['is_touching']]
if touching:
    centroids_x = [tp['centroid'][0] for tp in touching]
    centroids_y = [tp['centroid'][1] for tp in touching]
    leads_x = [tp['leading_edge'][0] for tp in touching]
    leads_y = [tp['leading_edge'][1] for tp in touching]
    frames_touching = [tp['frame'] for tp in touching]
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(touching)))
    
    # Draw trajectory line
    ax.plot(centroids_x, centroids_y, 'g--', alpha=0.7, linewidth=2, zorder=4)
    
    for j, (tp, color) in enumerate(zip(touching, colors)):
        cx, cy = tp['centroid']
        lx, ly = tp['leading_edge']
        
        marker_size = 200 if tp['is_new_swipe'] else 100
        edge_color = 'magenta' if tp['is_new_swipe'] else 'white'
        
        ax.scatter(cx, cy, c=[color], s=marker_size, marker='o', edgecolors=edge_color, 
                   linewidth=2, zorder=5)
        ax.scatter(lx, ly, c=[color], s=150, marker='*', edgecolors='white', 
                   linewidth=1, zorder=6)
        
        label = f"F{tp['frame']}"
        if tp['is_new_swipe']:
            label += " ★"
        ax.annotate(label, (cx, cy), textcoords="offset points", 
                    xytext=(10, -10), fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

ax.set_title('Extracted Touch Trajectory\n(★ = new swipe detected, magenta border)', 
             fontsize=14, fontweight='bold')
ax.legend(['Trajectory'], loc='lower left', fontsize=10)
ax.axis('off')
plt.tight_layout()
plt.savefig('/home/claude/v2_viz5_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ 6: Trace area over time chart ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Trace Metrics Over Time', fontsize=14, fontweight='bold')

# Trace area per frame
trace_areas = [np.sum(m > 0) for m in trace_masks]
ax1.bar(range(1, len(trace_areas)+1), trace_areas, color='orange', alpha=0.7)
ax1.set_xlabel('Frame')
ax1.set_ylabel('Orange Trace Pixels')
ax1.set_title('Total Trace Area Per Frame')
ax1.axhline(y=2000, color='red', linestyle='--', alpha=0.5, label='New swipe threshold')
ax1.legend()

# New pixels per transition
new_pixel_counts = [tp.get('new_pixels', 0) for tp in touch_points]
disappeared_counts = [tp.get('disappeared_pixels', 0) for tp in touch_points]
x = [tp['frame'] for tp in touch_points]
ax2.bar(x, new_pixel_counts, color='green', alpha=0.7, label='New orange pixels')
ax2.bar(x, [-d for d in disappeared_counts], color='red', alpha=0.5, label='Disappeared pixels')
ax2.set_xlabel('Frame')
ax2.set_ylabel('Pixel Count')
ax2.set_title('New vs Disappeared Trace Pixels Per Frame Transition')
ax2.legend()
ax2.axhline(y=0, color='black', linewidth=0.5)

# Mark new swipe frames
for tp in touch_points:
    if tp.get('is_new_swipe', False):
        ax2.axvline(x=tp['frame'], color='magenta', linestyle='--', alpha=0.7)
        ax2.text(tp['frame'], max(new_pixel_counts) * 0.9, '★', fontsize=14, 
                 ha='center', color='magenta')

plt.tight_layout()
plt.savefig('/home/claude/v2_viz6_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll visualizations saved!")
print(f"\nSummary: {len(touching)} frames with detected touch out of {len(touch_points)} transitions")
new_swipes = [tp for tp in touch_points if tp.get('is_new_swipe', False)]
print(f"New swipe events detected: {len(new_swipes)} at frames {[tp['frame'] for tp in new_swipes]}")

# Save data
output_data = {
    'total_frames': len(frames_rgb),
    'fps': 60,
    'trick': 'Nollie 360 Double Flip',
    'touch_points': touch_points,
    'trace_areas': trace_areas,
}
with open('/home/claude/v2_touch_data.json', 'w') as f:
    json.dump(output_data, f, indent=2)
