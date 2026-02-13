import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load batch 4 frames
frame_dir = Path("/mnt/user-data/uploads")
frame_files = sorted(frame_dir.glob("img_*.jpg"))
print(f"Found {len(frame_files)} frames")

frames_bgr = [cv2.imread(str(f)) for f in frame_files]
frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
frames_hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames_bgr]
h, w = frames_gray[0].shape
print(f"Frame size: {w}x{h}")

# HUD mask (same as previous batches)
hud_mask = np.ones((h, w), dtype=np.uint8) * 255
hud_mask[:420, :] = 0    # top HUD
hud_mask[1500:, :] = 0   # bottom HUD
hud_mask[:, :70] = 0     # left edge

# ============================================================
# STAGE 1: Frame diffs + duplicate detection
# ============================================================
print("\n=== Frame Diffs ===")
diffs = []
unique_flags = [True]  # first frame is always "unique"
for i in range(1, len(frames_gray)):
    diff = cv2.absdiff(frames_gray[i], frames_gray[i-1])
    diff_masked = cv2.bitwise_and(diff, hud_mask)
    diffs.append(diff_masked)
    mean_val = diff_masked.mean()
    is_dup = mean_val < 0.5
    unique_flags.append(not is_dup)
    tag = " [DUP]" if is_dup else ""
    print(f"  {i:2d}→{i+1:2d}: mean={mean_val:.2f}, max={diff_masked.max()}{tag}")

unique_count = sum(unique_flags)
print(f"\nUnique frames: {unique_count}/{len(frames_gray)}")

# ============================================================
# STAGE 2: Trace isolation with broadened HSV range
# ============================================================
print("\n=== Trace Isolation ===")
trace_masks = []
for i, hsv in enumerate(frames_hsv):
    # Orange/fire trace: H 0-35, moderate saturation, bright
    lower = np.array([0, 50, 120])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_and(mask, hud_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    trace_masks.append(mask)
    pc = np.sum(mask > 0)
    if pc > 50:
        coords = np.argwhere(mask > 0)
        cy, cx = coords.mean(axis=0)
        miny, maxy = coords[:, 0].min(), coords[:, 0].max()
        minx, maxx = coords[:, 1].min(), coords[:, 1].max()
        print(f"  Frame {i+1:2d}: {pc:6d} px, centroid=({cx:.0f},{cy:.0f}), bbox=({minx},{miny})-({maxx},{maxy})")
    else:
        print(f"  Frame {i+1:2d}: {pc:6d} px (minimal)")

# ============================================================
# STAGE 3: Connected component analysis with shape metrics
# ============================================================
print("\n=== Connected Components ===")
blob_data = []
for i, mask in enumerate(trace_masks):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [(j, stats[j, cv2.CC_STAT_AREA], centroids[j],
              stats[j, cv2.CC_STAT_WIDTH], stats[j, cv2.CC_STAT_HEIGHT])
             for j in range(1, num_labels) if stats[j, cv2.CC_STAT_AREA] > 50]
    blobs.sort(key=lambda x: -x[1])
    
    blob_info = []
    for j, (label, area, centroid, bw, bh) in enumerate(blobs):
        aspect_ratio = bw / max(bh, 1)
        blob_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        else:
            circularity = 0
        compactness = area / max(bw * bh, 1)
        
        blob_info.append({
            'label': label, 'area': area,
            'centroid': (float(centroid[0]), float(centroid[1])),
            'width': bw, 'height': bh,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'compactness': compactness,
        })
    
    blob_data.append(blob_info)
    if blob_info:
        main = blob_info[0]
        shape = "CIRCULAR" if main['circularity'] > 0.5 else "ELONGATED" if main['aspect_ratio'] > 2 or main['aspect_ratio'] < 0.5 else "IRREGULAR"
        print(f"  Frame {i+1:2d}: {len(blob_info)} blobs, main={main['area']}px "
              f"circ={main['circularity']:.2f} AR={main['aspect_ratio']:.2f} comp={main['compactness']:.2f} → {shape}")
    else:
        print(f"  Frame {i+1:2d}: no blobs")

# ============================================================
# STAGE 4: Touch classification (refined for hold + swipe + push)
# ============================================================
print("\n=== Touch Classification ===")
touch_results = []

# Also classify frame 1 (no transition, just static analysis)
for i in range(len(frames_gray)):
    cur_mask = trace_masks[i]
    cur_area = int(np.sum(cur_mask > 0))
    blobs = blob_data[i]
    main_blob = blobs[0] if blobs else None
    
    # Brightness-based touch point
    hsv = frames_hsv[i]
    v_ch = hsv[:, :, 2].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    brightness = (v_ch * s_ch / 255.0)
    brightness[cur_mask == 0] = 0
    bright_point = None
    if brightness.max() > 0:
        max_loc = np.unravel_index(brightness.argmax(), brightness.shape)
        bright_point = (int(max_loc[1]), int(max_loc[0]))
    
    if i > 0:
        diff = diffs[i-1]
        prev_mask = trace_masks[i-1]
        new_trace = cv2.bitwise_and(cur_mask, cv2.bitwise_not(prev_mask))
        new_count = int(np.sum(new_trace > 0))
        gone = cv2.bitwise_and(prev_mask, cv2.bitwise_not(cur_mask))
        gone_count = int(np.sum(gone > 0))
        prev_area = int(np.sum(prev_mask > 0))
        area_delta = cur_area - prev_area
        
        # Leading edge
        diff_in_new = diff.copy()
        diff_in_new[new_trace == 0] = 0
        leading_edge = None
        if diff_in_new.max() > 0:
            max_loc = np.unravel_index(diff_in_new.argmax(), diff_in_new.shape)
            leading_edge = (int(max_loc[1]), int(max_loc[0]))
    else:
        new_count = cur_area
        gone_count = 0
        area_delta = cur_area
        prev_area = 0
        leading_edge = None
    
    # Classify
    touch_type = "none"
    centroid = None
    confidence = 0.0
    
    if main_blob and cur_area > 200:
        circ = main_blob['circularity']
        ar = main_blob['aspect_ratio']
        centroid = main_blob['centroid']
        
        if circ > 0.4 and 0.5 < ar < 2.0 and cur_area < 8000:
            touch_type = "hold"
            confidence = min(1.0, cur_area / 1000)
        elif i > 0 and new_count > 2000 and area_delta > prev_area * 0.3:
            touch_type = "new_swipe"
            confidence = min(1.0, new_count / 5000)
        elif i > 0 and new_count > 500 and cur_area > 2000:
            touch_type = "swipe"
            confidence = min(1.0, new_count / 2000)
        elif i > 0 and cur_area > 500 and gone_count > new_count:
            touch_type = "fading"
            confidence = min(1.0, cur_area / 5000)
        elif cur_area > 200:
            touch_type = "hold" if circ > 0.3 else "residual"
            confidence = min(0.8, cur_area / 2000)
    elif cur_area > 50:
        touch_type = "residual"
        confidence = 0.2
    
    touch_results.append({
        'frame': i + 1,
        'touch_type': touch_type,
        'centroid': centroid,
        'bright_point': bright_point,
        'leading_edge': leading_edge,
        'cur_area': cur_area,
        'new_count': new_count,
        'gone_count': gone_count,
        'area_delta': area_delta,
        'confidence': confidence,
        'main_circularity': main_blob['circularity'] if main_blob else 0,
        'main_aspect_ratio': main_blob['aspect_ratio'] if main_blob else 0,
    })
    
    type_icons = {'hold': '●', 'swipe': '→', 'new_swipe': '★→', 'fading': '↓', 'residual': '·', 'none': '✕'}
    icon = type_icons.get(touch_type, '?')
    circ_str = f"circ={main_blob['circularity']:.2f} AR={main_blob['aspect_ratio']:.2f}" if main_blob else ""
    cent_str = f" at ({centroid[0]:.0f},{centroid[1]:.0f})" if centroid else ""
    print(f"  F{i+1:2d}: {icon} {touch_type:10s} area={cur_area:5d} new={new_count:5d} gone={gone_count:5d} "
          f"Δ={area_delta:+6d} {circ_str}{cent_str}")

# ============================================================
# VIZ A: All 20 frames with classification overlay
# ============================================================
print("\n=== Generating Visualizations ===")
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Batch 4: Hold + Push Swipe Analysis (Score 903)', fontsize=16, fontweight='bold')

type_colors = {'hold': 'cyan', 'swipe': 'lime', 'new_swipe': 'magenta', 
               'fading': 'yellow', 'residual': 'gray', 'none': 'red'}

for i in range(20):
    ax = axes[i // 5][i % 5]
    overlay = frames_rgb[i].copy()
    trace_overlay = overlay.copy()
    trace_overlay[trace_masks[i] > 0] = [255, 120, 0]
    overlay = cv2.addWeighted(overlay, 0.6, trace_overlay, 0.4, 0)
    ax.imshow(overlay)
    
    tr = touch_results[i]
    color = type_colors.get(tr['touch_type'], 'white')
    
    if tr['centroid']:
        cx, cy = tr['centroid']
        ax.plot(cx, cy, 'o', color=color, markersize=12, markeredgecolor='white', markeredgewidth=1.5)
    
    if tr['bright_point']:
        bx, by = tr['bright_point']
        ax.plot(bx, by, 's', color='cyan', markersize=8, markeredgecolor='white', markeredgewidth=1)
    
    if tr['leading_edge'] and tr['touch_type'] in ('swipe', 'new_swipe'):
        lx, ly = tr['leading_edge']
        ax.plot(lx, ly, '*', color='red', markersize=12, markeredgecolor='white', markeredgewidth=1)
    
    ax.set_title(f'F{i+1} | {tr["touch_type"]} ({tr["cur_area"]}px)\n'
                 f'circ={tr["main_circularity"]:.2f} AR={tr["main_aspect_ratio"]:.2f}', 
                 fontsize=8, color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/b4_all_frames.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved b4_all_frames.png")

# ============================================================
# VIZ B: Hold region zoom (frames 1-6) + swipe emergence
# ============================================================
fig, axes = plt.subplots(3, 6, figsize=(30, 15))
fig.suptitle('Hold → Swipe Transition Detail (Frames 1-6)', fontsize=14, fontweight='bold')

for col, fi in enumerate(range(6)):
    # Row 0: Original with trace overlay
    ax = axes[0][col]
    overlay = frames_rgb[fi].copy()
    trace_overlay = overlay.copy()
    trace_overlay[trace_masks[fi] > 0] = [255, 120, 0]
    overlay = cv2.addWeighted(overlay, 0.5, trace_overlay, 0.5, 0)
    
    tr = touch_results[fi]
    if tr['centroid']:
        cx, cy = tr['centroid']
        color = type_colors.get(tr['touch_type'], 'white')
        cv2.circle(overlay, (int(cx), int(cy)), 15, (255, 255, 255), 2)
    if tr['bright_point']:
        bx, by = tr['bright_point']
        cv2.drawMarker(overlay, (bx, by), (0, 255, 255), cv2.MARKER_SQUARE, 12, 2)
    
    ax.imshow(overlay)
    ax.set_title(f'F{fi+1}: {tr["touch_type"]} ({tr["cur_area"]}px)', fontsize=9,
                 color=type_colors.get(tr['touch_type'], 'white'))
    ax.axis('off')
    
    # Row 1: Trace mask with components colored
    ax = axes[1][col]
    mask = trace_masks[fi]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    comp_vis = np.zeros((h, w, 3), dtype=np.uint8)
    cmap = plt.cm.Set1(np.linspace(0, 1, max(num_labels, 2)))
    for lbl in range(1, num_labels):
        if stats[lbl, cv2.CC_STAT_AREA] > 20:
            c = (np.array(cmap[lbl % len(cmap)][:3]) * 255).astype(np.uint8)
            comp_vis[labels == lbl] = c
    ax.imshow(comp_vis)
    n_blobs = len([1 for lbl in range(1, num_labels) if stats[lbl, cv2.CC_STAT_AREA] > 20])
    ax.set_title(f'{n_blobs} blobs', fontsize=9)
    ax.axis('off')
    
    # Row 2: New pixels (red) vs persisting (green) vs disappeared (blue)
    ax = axes[2][col]
    if fi > 0:
        new_trace = cv2.bitwise_and(trace_masks[fi], cv2.bitwise_not(trace_masks[fi-1]))
        persist = cv2.bitwise_and(trace_masks[fi], trace_masks[fi-1])
        gone = cv2.bitwise_and(trace_masks[fi-1], cv2.bitwise_not(trace_masks[fi]))
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        composite[new_trace > 0] = [255, 0, 0]
        composite[persist > 0] = [0, 255, 0]
        composite[gone > 0] = [0, 0, 255]
        ax.imshow(composite)
        nc = np.sum(new_trace > 0)
        pc = np.sum(persist > 0)
        gc = np.sum(gone > 0)
        ax.set_title(f'R:new={nc} G:persist={pc}\nB:gone={gc}', fontsize=7)
    else:
        ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))
        ax.set_title('(first frame)', fontsize=7)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/b4_hold_zoom.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved b4_hold_zoom.png")

# ============================================================
# VIZ C: Swipe phase (frames 7-14) with push trace
# ============================================================
fig, axes = plt.subplots(3, 8, figsize=(40, 15))
fig.suptitle('Active Swipe / Push Phase (Frames 7-14)', fontsize=14, fontweight='bold')

for col, fi in enumerate(range(6, 14)):
    # Row 0: overlay
    ax = axes[0][col]
    overlay = frames_rgb[fi].copy()
    trace_overlay = overlay.copy()
    trace_overlay[trace_masks[fi] > 0] = [255, 120, 0]
    overlay = cv2.addWeighted(overlay, 0.5, trace_overlay, 0.5, 0)
    tr = touch_results[fi]
    if tr['bright_point']:
        bx, by = tr['bright_point']
        cv2.drawMarker(overlay, (bx, by), (0, 255, 255), cv2.MARKER_SQUARE, 15, 2)
    if tr['leading_edge'] and tr['touch_type'] in ('swipe', 'new_swipe'):
        lx, ly = tr['leading_edge']
        cv2.drawMarker(overlay, (lx, ly), (255, 0, 255), cv2.MARKER_STAR, 15, 2)
    ax.imshow(overlay)
    ax.set_title(f'F{fi+1}: {tr["touch_type"]}', fontsize=9,
                 color=type_colors.get(tr['touch_type'], 'white'))
    ax.axis('off')
    
    # Row 1: trace mask
    ax = axes[1][col]
    ax.imshow(trace_masks[fi], cmap='hot')
    ax.set_title(f'{tr["cur_area"]}px', fontsize=9)
    ax.axis('off')
    
    # Row 2: new/persist/gone
    ax = axes[2][col]
    new_trace = cv2.bitwise_and(trace_masks[fi], cv2.bitwise_not(trace_masks[fi-1]))
    persist = cv2.bitwise_and(trace_masks[fi], trace_masks[fi-1])
    gone = cv2.bitwise_and(trace_masks[fi-1], cv2.bitwise_not(trace_masks[fi]))
    composite = np.zeros((h, w, 3), dtype=np.uint8)
    composite[new_trace > 0] = [255, 0, 0]
    composite[persist > 0] = [0, 255, 0]
    composite[gone > 0] = [0, 0, 255]
    ax.imshow(composite)
    nc = np.sum(new_trace > 0)
    ax.set_title(f'new={nc}', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/b4_swipe_phase.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved b4_swipe_phase.png")

# ============================================================
# VIZ D: Metrics timeline
# ============================================================
fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
fig.suptitle('Batch 4: Temporal Metrics', fontsize=14, fontweight='bold')

frames_idx = [tr['frame'] for tr in touch_results]
areas = [tr['cur_area'] for tr in touch_results]
new_counts = [tr['new_count'] for tr in touch_results]
gone_counts = [tr['gone_count'] for tr in touch_results]
circularities = [tr['main_circularity'] for tr in touch_results]
aspect_ratios = [tr['main_aspect_ratio'] for tr in touch_results]

# Plot 0: Trace area
ax = axes[0]
ax.fill_between(frames_idx, areas, alpha=0.3, color='orange')
ax.plot(frames_idx, areas, 'o-', color='orange', markersize=6)
for tr in touch_results:
    if tr['touch_type'] in ('new_swipe', 'hold'):
        ax.axvline(tr['frame'], color='red' if tr['touch_type'] == 'new_swipe' else 'cyan', 
                   alpha=0.3, linestyle='--')
ax.set_ylabel('Trace Area (px)')
ax.set_title('Trace Area Over Time')
# Label touch types
for tr in touch_results:
    color = type_colors.get(tr['touch_type'], 'gray')
    ax.annotate(tr['touch_type'][:4], (tr['frame'], tr['cur_area']),
               fontsize=6, color=color, ha='center', va='bottom')

# Plot 1: New vs Gone pixels
ax = axes[1]
ax.bar(frames_idx, new_counts, alpha=0.7, color='red', label='New pixels', width=0.4)
ax.bar([f+0.4 for f in frames_idx], gone_counts, alpha=0.7, color='blue', label='Gone pixels', width=0.4)
ax.set_ylabel('Pixel Count')
ax.legend(fontsize=8)
ax.set_title('New vs Disappeared Pixels')

# Plot 2: Circularity
ax = axes[2]
ax.plot(frames_idx, circularities, 'o-', color='cyan', label='Circularity')
ax.axhline(0.4, color='cyan', alpha=0.3, linestyle='--', label='Hold threshold')
ax.set_ylabel('Circularity')
ax.legend(fontsize=8)
ax.set_title('Main Blob Circularity')

# Plot 3: Aspect ratio
ax = axes[3]
ax.plot(frames_idx, aspect_ratios, 'o-', color='magenta', label='Aspect Ratio')
ax.axhline(1.0, color='gray', alpha=0.3, linestyle='--')
ax.set_ylabel('Aspect Ratio (W/H)')
ax.set_xlabel('Frame')
ax.legend(fontsize=8)
ax.set_title('Main Blob Aspect Ratio (1.0 = square)')

plt.tight_layout()
plt.savefig('/home/claude/b4_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved b4_metrics.png")

# ============================================================
# VIZ E: Shape analysis - circularity vs area scatter
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Shape Feature Space: Hold vs Swipe vs Fade', fontsize=14, fontweight='bold')

# Scatter: area vs circularity
ax = axes[0]
for tr in touch_results:
    if tr['cur_area'] > 50:
        color = type_colors.get(tr['touch_type'], 'gray')
        ax.scatter(tr['cur_area'], tr['main_circularity'], c=color, s=80, edgecolors='white', zorder=5)
        ax.annotate(f'F{tr["frame"]}', (tr['cur_area'], tr['main_circularity']), fontsize=7,
                   ha='left', va='bottom')

ax.set_xlabel('Trace Area (px)')
ax.set_ylabel('Circularity')
ax.axhline(0.4, color='cyan', alpha=0.3, linestyle='--', label='Hold threshold')
ax.set_title('Area vs Circularity')
ax.legend()

# Scatter: area vs aspect ratio
ax = axes[1]
for tr in touch_results:
    if tr['cur_area'] > 50:
        color = type_colors.get(tr['touch_type'], 'gray')
        ax.scatter(tr['cur_area'], tr['main_aspect_ratio'], c=color, s=80, edgecolors='white', zorder=5)
        ax.annotate(f'F{tr["frame"]}', (tr['cur_area'], tr['main_aspect_ratio']), fontsize=7,
                   ha='left', va='bottom')

ax.set_xlabel('Trace Area (px)')
ax.set_ylabel('Aspect Ratio')
ax.axhline(1.0, color='gray', alpha=0.3, linestyle='--')
ax.set_title('Area vs Aspect Ratio')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items() if t != 'none']
fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('/home/claude/b4_shape_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved b4_shape_analysis.png")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("BATCH 4 SUMMARY")
print("="*60)
print(f"Total frames: {len(frames_gray)}, Unique: {unique_count}")
print(f"\nClassification breakdown:")
from collections import Counter
type_counts = Counter(tr['touch_type'] for tr in touch_results)
for t, c in type_counts.most_common():
    frames_list = [tr['frame'] for tr in touch_results if tr['touch_type'] == t]
    print(f"  {t:12s}: {c:2d} frames → {frames_list}")

print(f"\nKey observations:")
# Find hold-to-swipe transition
for i in range(1, len(touch_results)):
    prev_type = touch_results[i-1]['touch_type']
    curr_type = touch_results[i]['touch_type']
    if prev_type != curr_type:
        print(f"  F{touch_results[i-1]['frame']}→F{touch_results[i]['frame']}: {prev_type} → {curr_type}")

print("\nDone!")
