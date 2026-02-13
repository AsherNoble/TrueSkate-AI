import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load frames
frame_dir = Path("/mnt/user-data/uploads")
frame_files = sorted([f for f in frame_dir.glob("img_*.jpg") if int(f.stem.split('_')[1]) <= 20])
print(f"Found {len(frame_files)} frames")

frames_bgr = [cv2.imread(str(f)) for f in frame_files]
frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
frames_hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames_bgr]
h, w = frames_gray[0].shape

# ============================================================
# HUD MASK: exclude UI elements that contain warm colors
# ============================================================
hud_mask = np.ones((h, w), dtype=np.uint8) * 255
# Top bar (score, timer, trick labels, banner)
hud_mask[:420, :] = 0
# Bottom bar (speed, height, Switch label)
hud_mask[1500:, :] = 0
# Left side buttons (arrows, camera)
hud_mask[:, :70] = 0

print(f"HUD mask excludes {np.sum(hud_mask == 0)} pixels ({100*np.sum(hud_mask==0)/(h*w):.1f}%)")

# ============================================================
# Trace detection with HUD masking
# ============================================================
print("\n=== Trace Detection (HUD-masked) ===")

trace_masks = []
for i, hsv in enumerate(frames_hsv):
    lower1 = np.array([0, 50, 120])
    upper1 = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1)
    
    # Apply HUD mask
    mask = cv2.bitwise_and(mask, hud_mask)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    trace_masks.append(mask)
    pc = np.sum(mask > 0)
    if pc > 50:
        coords = np.argwhere(mask > 0)
        cy, cx = coords.mean(axis=0)
        print(f"  Frame {i+1:2d}: {pc:6d} trace px, centroid=({cx:.0f},{cy:.0f})")
    else:
        print(f"  Frame {i+1:2d}: {pc:6d} trace px (minimal)")

# ============================================================
# Largest blob tracking + leading edge via brightness
# ============================================================
print("\n=== Largest Blob + Brightness Leading Edge ===")

blob_data = []
for i, (mask, hsv) in enumerate(zip(trace_masks, frames_hsv)):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Find largest blob (excluding background)
    blobs = [(j, stats[j, cv2.CC_STAT_AREA], centroids[j]) 
             for j in range(1, num_labels) if stats[j, cv2.CC_STAT_AREA] > 50]
    blobs.sort(key=lambda x: -x[1])  # Sort by area descending
    
    if blobs:
        main_label, main_area, main_centroid = blobs[0]
        
        # Create mask for just the main blob
        main_mask = (labels == main_label).astype(np.uint8) * 255
        
        # Find brightest point in main blob (V channel in HSV)
        v_channel = hsv[:, :, 2].copy().astype(np.float32)
        s_channel = hsv[:, :, 1].copy().astype(np.float32)
        # Weight by saturation * value — the trace glow has both high S and V
        brightness = (v_channel * s_channel / 255.0)
        brightness[main_mask == 0] = 0
        
        if brightness.max() > 0:
            max_loc = np.unravel_index(brightness.argmax(), brightness.shape)
            bright_y, bright_x = max_loc
        else:
            bright_x, bright_y = main_centroid
        
        blob_data.append({
            'frame': i + 1,
            'main_area': int(main_area),
            'main_centroid': (float(main_centroid[0]), float(main_centroid[1])),
            'brightest': (int(bright_x), int(bright_y)),
            'num_blobs': len(blobs),
            'all_areas': [int(b[1]) for b in blobs[:5]],
        })
        
        extra_info = f"  +{len(blobs)-1} more" if len(blobs) > 1 else ""
        print(f"  Frame {i+1:2d}: main blob={main_area:5d}px at ({main_centroid[0]:.0f},{main_centroid[1]:.0f}), "
              f"brightest=({bright_x},{bright_y}){extra_info}")
    else:
        blob_data.append({
            'frame': i + 1, 'main_area': 0, 'main_centroid': None,
            'brightest': None, 'num_blobs': 0, 'all_areas': [],
        })
        print(f"  Frame {i+1:2d}: no trace blob")

# ============================================================
# Diff-based leading edge (only within main blob region)
# ============================================================
print("\n=== Diff-Based Touch Point (HUD-masked) ===")

touch_results = []
for i in range(1, len(frames_gray)):
    diff = cv2.absdiff(frames_gray[i], frames_gray[i-1])
    diff = cv2.bitwise_and(diff, hud_mask)
    
    cur_mask = trace_masks[i]
    prev_mask = trace_masks[i-1]
    
    # NEW trace pixels only
    new_trace = cv2.bitwise_and(cur_mask, cv2.bitwise_not(prev_mask))
    
    # Diff within current trace
    diff_thresh = (diff > 15).astype(np.uint8) * 255
    diff_in_trace = cv2.bitwise_and(diff_thresh, cur_mask)
    
    # Combined
    combined = cv2.bitwise_or(new_trace, diff_in_trace)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    # Find largest connected component in combined signal
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
    blobs = [(j, stats[j, cv2.CC_STAT_AREA], centroids[j]) 
             for j in range(1, num_labels) if stats[j, cv2.CC_STAT_AREA] > 30]
    blobs.sort(key=lambda x: -x[1])
    
    new_trace_area = int(np.sum(new_trace > 0))
    disappeared = int(np.sum(cv2.bitwise_and(prev_mask, cv2.bitwise_not(cur_mask)) > 0))
    
    # Detect new swipe: sudden large increase in trace area
    cur_area = blob_data[i]['main_area'] if blob_data[i]['main_centroid'] else 0
    prev_area = blob_data[i-1]['main_area'] if blob_data[i-1]['main_centroid'] else 0
    area_jump = cur_area - prev_area
    
    # A "new swipe" = area jump > 50% of previous area AND new trace pixels > 5000
    is_new_swipe = (area_jump > prev_area * 0.5 and new_trace_area > 5000) if prev_area > 500 else (new_trace_area > 5000)
    
    if blobs:
        _, largest_area, largest_centroid = blobs[0]
        cx, cy = largest_centroid
        
        # Leading edge: brightest diff point in the combined signal
        masked_diff = diff.copy()
        masked_diff[combined == 0] = 0
        if masked_diff.max() > 0:
            max_loc = np.unravel_index(masked_diff.argmax(), masked_diff.shape)
            lead_y, lead_x = max_loc
        else:
            lead_x, lead_y = cx, cy
        
        touch_results.append({
            'frame': i + 1,
            'touching': True,
            'centroid': (float(cx), float(cy)),
            'leading_edge': (int(lead_x), int(lead_y)),
            'new_trace_area': new_trace_area,
            'disappeared': disappeared,
            'area_jump': int(area_jump),
            'is_new_swipe': bool(is_new_swipe),
        })
        marker = " ★ NEW SWIPE" if is_new_swipe else ""
        print(f"  {i:2d}→{i+1:2d}: centroid=({cx:.0f},{cy:.0f}) leading=({lead_x},{lead_y}) "
              f"new={new_trace_area} Δarea={area_jump:+d}{marker}")
    else:
        touch_results.append({
            'frame': i + 1, 'touching': False, 'centroid': None,
            'leading_edge': None, 'new_trace_area': new_trace_area,
            'disappeared': disappeared, 'area_jump': int(area_jump),
            'is_new_swipe': False,
        })
        print(f"  {i:2d}→{i+1:2d}: no signal (new={new_trace_area}, disappeared={disappeared})")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n=== Generating Visualizations ===")

# --- VIZ A: Side-by-side all 20 frames with trace overlay + touch points ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Nollie 360 Double Flip — Touch Extraction (HUD-masked)', fontsize=16, fontweight='bold')

for i in range(20):
    ax = axes[i // 5][i % 5]
    overlay = frames_rgb[i].copy()
    
    # Draw trace in semi-transparent orange
    trace_overlay = overlay.copy()
    trace_overlay[trace_masks[i] > 0] = [255, 120, 0]
    overlay = cv2.addWeighted(overlay, 0.6, trace_overlay, 0.4, 0)
    
    ax.imshow(overlay)
    
    # Plot touch point from diff analysis
    if i > 0:
        tr = touch_results[i - 1]
        if tr['touching']:
            cx, cy = tr['centroid']
            lx, ly = tr['leading_edge']
            color = 'magenta' if tr['is_new_swipe'] else 'lime'
            ax.plot(cx, cy, 'o', color=color, markersize=10, markeredgecolor='white', markeredgewidth=1.5)
            ax.plot(lx, ly, '*', color='red', markersize=12, markeredgecolor='white', markeredgewidth=1)
    
    # Plot brightness-based point
    bd = blob_data[i]
    if bd['brightest']:
        bx, by = bd['brightest']
        ax.plot(bx, by, 's', color='cyan', markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    # Title
    area = blob_data[i]['main_area']
    swipe_marker = " ★" if (i > 0 and touch_results[i-1].get('is_new_swipe', False)) else ""
    ax.set_title(f'F{i+1} | trace={area}px{swipe_marker}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/v2r_all_frames.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ B: Trajectory plot ---
fig, ax = plt.subplots(1, 1, figsize=(10, 16))
ax.imshow(frames_rgb[5], alpha=0.4)  # frame 6 as background (peak trace)

# Plot diff-based centroids
touching = [tr for tr in touch_results if tr['touching']]
if touching:
    cx_list = [tr['centroid'][0] for tr in touching]
    cy_list = [tr['centroid'][1] for tr in touching]
    frames_list = [tr['frame'] for tr in touching]
    colors = plt.cm.viridis(np.linspace(0, 1, len(touching)))
    
    ax.plot(cx_list, cy_list, '--', color='lime', alpha=0.6, linewidth=2)
    
    for tr, color in zip(touching, colors):
        cx, cy = tr['centroid']
        lx, ly = tr['leading_edge']
        
        ec = 'magenta' if tr['is_new_swipe'] else 'white'
        sz = 200 if tr['is_new_swipe'] else 100
        
        ax.scatter(cx, cy, c=[color], s=sz, marker='o', edgecolors=ec, linewidth=2, zorder=5)
        ax.scatter(lx, ly, c=[color], s=120, marker='*', edgecolors='white', linewidth=1, zorder=6)
        
        label = f"F{tr['frame']}"
        if tr['is_new_swipe']: label += "★"
        ax.annotate(label, (cx, cy), textcoords="offset points",
                    xytext=(10, -10), fontsize=8, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))

# Also plot brightness-based points as a separate series
bright_x = [bd['brightest'][0] for bd in blob_data if bd['brightest']]
bright_y = [bd['brightest'][1] for bd in blob_data if bd['brightest']]
bright_f = [bd['frame'] for bd in blob_data if bd['brightest']]
ax.plot(bright_x, bright_y, 's-', color='cyan', alpha=0.5, markersize=6, linewidth=1, label='Brightness peak')

ax.set_title('Touch Trajectory: Diff Centroid (circles) + Leading Edge (stars) + Brightness (squares)', 
             fontsize=11, fontweight='bold')
ax.legend(loc='lower left')
ax.axis('off')
plt.tight_layout()
plt.savefig('/home/claude/v2r_trajectory.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ C: Metrics timeline ---
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Trace Metrics Timeline — Nollie 360 Double Flip', fontsize=14, fontweight='bold')

frames_x = list(range(1, 21))

# Trace area
areas = [bd['main_area'] for bd in blob_data]
axes[0].fill_between(frames_x, areas, alpha=0.3, color='orange')
axes[0].plot(frames_x, areas, 'o-', color='orange', markersize=6)
axes[0].set_ylabel('Main Blob Area (px)')
axes[0].set_title('Trace Area Over Time')
# Mark new swipes
for tr in touch_results:
    if tr.get('is_new_swipe'):
        axes[0].axvline(x=tr['frame'], color='magenta', linestyle='--', alpha=0.7)

# New vs disappeared
new_areas = [0] + [tr['new_trace_area'] for tr in touch_results]
disappeared = [0] + [tr['disappeared'] for tr in touch_results]
axes[1].bar(frames_x, new_areas, color='green', alpha=0.7, label='New trace pixels')
axes[1].bar(frames_x, [-d for d in disappeared], color='red', alpha=0.5, label='Disappeared')
axes[1].set_ylabel('Pixel Count')
axes[1].set_title('New vs Disappeared Trace Pixels')
axes[1].legend()
axes[1].axhline(y=0, color='black', linewidth=0.5)

# Area jump (derivative)
area_jumps = [0] + [tr['area_jump'] for tr in touch_results]
colors_bar = ['magenta' if (i > 0 and touch_results[i-1].get('is_new_swipe')) else 'steelblue' 
              for i in range(20)]
axes[2].bar(frames_x, area_jumps, color=colors_bar, alpha=0.7)
axes[2].set_ylabel('Area Change')
axes[2].set_xlabel('Frame')
axes[2].set_title('Trace Area Jump (Δ) — Magenta = New Swipe Detected')
axes[2].axhline(y=0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig('/home/claude/v2r_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ D: Critical overlap frames 4-8, zoomed ---
fig, axes = plt.subplots(3, 5, figsize=(25, 15))
fig.suptitle('Overlap Analysis: Frames 4-8 (Swipe Transition Zone)', fontsize=14, fontweight='bold')

for col, fi in enumerate(range(3, 8)):  # frames 4-8 (0-indexed 3-7)
    # Row 0: Original
    axes[0][col].imshow(frames_rgb[fi])
    axes[0][col].set_title(f'Frame {fi+1}', fontsize=10)
    axes[0][col].axis('off')
    
    # Row 1: Trace mask colored by component
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    mask = trace_masks[fi]
    num_labels, labels, stats, centroids_cc = cv2.connectedComponentsWithStats(mask)
    cmap_colors = [(255,100,0), (0,255,100), (100,100,255), (255,255,0), (255,0,255)]
    for j in range(1, num_labels):
        if stats[j, cv2.CC_STAT_AREA] > 50:
            c = cmap_colors[(j-1) % len(cmap_colors)]
            overlay[labels == j] = c
    axes[1][col].imshow(overlay)
    axes[1][col].set_title(f'Trace blobs (HUD-masked)', fontsize=9)
    axes[1][col].axis('off')
    
    # Row 2: New trace pixels (cyan) vs existing (green)
    composite = np.zeros((h, w, 3), dtype=np.uint8)
    if fi > 0:
        prev_mask = trace_masks[fi - 1]
        existing = cv2.bitwise_and(mask, prev_mask)
        new_only = cv2.bitwise_and(mask, cv2.bitwise_not(prev_mask))
        gone = cv2.bitwise_and(prev_mask, cv2.bitwise_not(mask))
        
        composite[:, :, 1] = existing  # green = persisting
        composite[:, :, 0] = new_only   # red = new
        # Blue channel: diff intensity in trace region
        diff = cv2.absdiff(frames_gray[fi], frames_gray[fi-1])
        diff_masked = cv2.bitwise_and(diff, mask)
        composite[:, :, 2] = np.clip(diff_masked * 2, 0, 255)
        
        new_count = np.sum(new_only > 0)
        gone_count = np.sum(gone > 0)
        axes[2][col].set_title(f'Red=new({new_count}) Green=persist Blue=diff', fontsize=8)
    axes[2][col].imshow(composite)
    axes[2][col].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/v2r_overlap_zoom.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nDone! All refined visualizations saved.")
new_swipes = [tr for tr in touch_results if tr.get('is_new_swipe')]
print(f"New swipe events: {len(new_swipes)} at frames {[tr['frame'] for tr in new_swipes]}")
