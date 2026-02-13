import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load batch 3 frames
frame_dir = Path("/mnt/user-data/uploads")
frame_files = sorted([f for f in frame_dir.glob("img_*.jpg") if int(f.stem.split('_')[1]) <= 20])
print(f"Found {len(frame_files)} frames")

frames_bgr = [cv2.imread(str(f)) for f in frame_files]
frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
frames_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames_bgr]
frames_hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in frames_bgr]
h, w = frames_gray[0].shape
print(f"Frame size: {w}x{h}")

# HUD mask
hud_mask = np.ones((h, w), dtype=np.uint8) * 255
hud_mask[:420, :] = 0
hud_mask[1500:, :] = 0
hud_mask[:, :70] = 0

# ============================================================
# STAGE 1: Basic diagnostics
# ============================================================
print("\n=== Frame Diffs ===")
diffs = []
for i in range(1, len(frames_gray)):
    diff = cv2.absdiff(frames_gray[i], frames_gray[i-1])
    diff_masked = cv2.bitwise_and(diff, hud_mask)
    diffs.append(diff_masked)
    mean_val = diff_masked.mean()
    print(f"  {i:2d}→{i+1:2d}: mean={mean_val:.2f}, max={diff_masked.max()}")

# ============================================================
# STAGE 2: Trace isolation (orange/fire hue, HUD-masked)
# ============================================================
print("\n=== Trace Isolation ===")
trace_masks = []
for i, hsv in enumerate(frames_hsv):
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
        print(f"  Frame {i+1:2d}: {pc:6d} px, centroid=({cx:.0f},{cy:.0f})")
    else:
        print(f"  Frame {i+1:2d}: {pc:6d} px (minimal)")

# ============================================================
# STAGE 3: Connected component analysis per frame
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
        # Circularity: how round is the blob?
        blob_mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
        else:
            circularity = 0
        
        # Compactness (area / bounding box area)
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
              f"circ={main['circularity']:.2f} AR={main['aspect_ratio']:.2f} → {shape}")
    else:
        print(f"  Frame {i+1:2d}: no blobs")

# ============================================================
# STAGE 4: Trace type classification per frame transition
# ============================================================
print("\n=== Touch Classification ===")

touch_results = []
for i in range(1, len(frames_gray)):
    diff = diffs[i-1]
    cur_mask = trace_masks[i]
    prev_mask = trace_masks[i-1]
    
    # New trace pixels
    new_trace = cv2.bitwise_and(cur_mask, cv2.bitwise_not(prev_mask))
    new_count = int(np.sum(new_trace > 0))
    
    # Disappeared trace pixels
    gone = cv2.bitwise_and(prev_mask, cv2.bitwise_not(cur_mask))
    gone_count = int(np.sum(gone > 0))
    
    # Current trace stats
    cur_area = int(np.sum(cur_mask > 0))
    prev_area = int(np.sum(prev_mask > 0))
    area_delta = cur_area - prev_area
    
    # Classify the main blob shape
    blobs = blob_data[i]
    main_blob = blobs[0] if blobs else None
    
    # Brightness-based touch point (within trace)
    hsv = frames_hsv[i]
    v_ch = hsv[:, :, 2].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    brightness = (v_ch * s_ch / 255.0)
    brightness[cur_mask == 0] = 0
    
    bright_point = None
    if brightness.max() > 0:
        max_loc = np.unravel_index(brightness.argmax(), brightness.shape)
        bright_point = (int(max_loc[1]), int(max_loc[0]))
    
    # Diff-based leading edge (new trace pixels only)
    diff_in_new = diff.copy()
    diff_in_new[new_trace == 0] = 0
    leading_edge = None
    if diff_in_new.max() > 0:
        max_loc = np.unravel_index(diff_in_new.argmax(), diff_in_new.shape)
        leading_edge = (int(max_loc[1]), int(max_loc[0]))
    
    # New trace blob shape analysis
    new_trace_clean = cv2.morphologyEx(new_trace, cv2.MORPH_OPEN, 
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    num_new, new_labels, new_stats, new_centroids = cv2.connectedComponentsWithStats(new_trace_clean)
    new_blobs = [(new_stats[j, cv2.CC_STAT_AREA], new_centroids[j],
                  new_stats[j, cv2.CC_STAT_WIDTH], new_stats[j, cv2.CC_STAT_HEIGHT])
                 for j in range(1, num_new) if new_stats[j, cv2.CC_STAT_AREA] > 20]
    new_blobs.sort(key=lambda x: -x[0])
    
    # Classify touch type
    touch_type = "none"
    centroid = None
    confidence = 0.0
    
    if main_blob and cur_area > 200:
        circ = main_blob['circularity']
        ar = main_blob['aspect_ratio']
        centroid = main_blob['centroid']
        
        if circ > 0.4 and 0.5 < ar < 2.0:
            # Roughly circular = HOLD
            touch_type = "hold"
            confidence = min(1.0, cur_area / 1000)
        elif new_count > 2000 and area_delta > prev_area * 0.3:
            # Large sudden appearance = NEW SWIPE
            touch_type = "new_swipe"
            confidence = min(1.0, new_count / 5000)
        elif new_count > 500 and cur_area > 2000:
            # Growing trace = ACTIVE SWIPE
            touch_type = "swipe"
            confidence = min(1.0, new_count / 2000)
        elif cur_area > 500 and gone_count > new_count:
            # Shrinking trace = FADING (finger lifted)
            touch_type = "fading"
            confidence = min(1.0, cur_area / 5000)
        elif cur_area > 200:
            # Some trace present but not clearly growing or shrinking
            touch_type = "hold" if circ > 0.3 else "residual"
            confidence = min(0.8, cur_area / 2000)
    elif cur_area > 50 and cur_area <= 200:
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
    extra = ""
    if centroid:
        extra = f" at ({centroid[0]:.0f},{centroid[1]:.0f})"
    print(f"  {i:2d}→{i+1:2d}: {icon} {touch_type:10s} area={cur_area:5d} new={new_count:5d} gone={gone_count:5d} "
          f"circ={main_blob['circularity']:.2f} AR={main_blob['aspect_ratio']:.2f}{extra}" if main_blob else f"{extra}")

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n=== Generating Visualizations ===")

# --- VIZ A: All 20 frames with type classification ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Batch 3: 360 Double Flip → 5-0 Grind (Hold + Swipe Detection)', fontsize=16, fontweight='bold')

type_colors = {'hold': 'cyan', 'swipe': 'lime', 'new_swipe': 'magenta', 
               'fading': 'yellow', 'residual': 'gray', 'none': 'red'}

for i in range(20):
    ax = axes[i // 5][i % 5]
    overlay = frames_rgb[i].copy()
    
    # Trace overlay
    trace_overlay = overlay.copy()
    trace_overlay[trace_masks[i] > 0] = [255, 120, 0]
    overlay = cv2.addWeighted(overlay, 0.6, trace_overlay, 0.4, 0)
    ax.imshow(overlay)
    
    if i > 0:
        tr = touch_results[i - 1]
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
                     f'circ={tr["main_circularity"]:.2f}', fontsize=8, color=color)
    else:
        area = int(np.sum(trace_masks[0] > 0))
        ax.set_title(f'F1 | {area}px', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/b3_all_frames.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ B: Trace shape analysis ---
fig, axes = plt.subplots(4, 5, figsize=(25, 20))
fig.suptitle('Trace Shape Analysis: Circularity & Aspect Ratio', fontsize=14, fontweight='bold')

for i, (mask, blobs) in enumerate(zip(trace_masks, blob_data)):
    ax = axes[i // 5][i % 5]
    
    # Show trace mask colored by blob
    display = np.zeros((h, w, 3), dtype=np.uint8)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    colors = [(255,100,0), (0,255,100), (100,100,255), (255,255,0), (255,0,255)]
    for j in range(1, num_labels):
        if stats[j, cv2.CC_STAT_AREA] > 50:
            c = colors[(j-1) % len(colors)]
            display[labels == j] = c
            
            # Draw bounding box
            x0 = stats[j, cv2.CC_STAT_LEFT]
            y0 = stats[j, cv2.CC_STAT_TOP]
            bw = stats[j, cv2.CC_STAT_WIDTH]
            bh = stats[j, cv2.CC_STAT_HEIGHT]
            cv2.rectangle(display, (x0, y0), (x0+bw, y0+bh), (255,255,255), 2)
    
    ax.imshow(display)
    
    if blobs:
        main = blobs[0]
        ax.set_title(f'F{i+1}: circ={main["circularity"]:.2f} AR={main["aspect_ratio"]:.2f}\n'
                     f'area={main["area"]} compact={main["compactness"]:.2f}', fontsize=8)
    else:
        ax.set_title(f'F{i+1}: no trace', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/b3_shape_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ C: Metrics timeline ---
fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
fig.suptitle('Batch 3 Metrics Timeline', fontsize=14, fontweight='bold')

frames_x = [tr['frame'] for tr in touch_results]

# Trace area
areas = [tr['cur_area'] for tr in touch_results]
axes[0].fill_between(frames_x, areas, alpha=0.3, color='orange')
axes[0].plot(frames_x, areas, 'o-', color='orange', markersize=6)
axes[0].set_ylabel('Trace Area (px)')
axes[0].set_title('Trace Area')

# Circularity
circs = [tr['main_circularity'] for tr in touch_results]
ars = [tr['main_aspect_ratio'] for tr in touch_results]
axes[1].plot(frames_x, circs, 'o-', color='cyan', label='Circularity')
axes[1].plot(frames_x, ars, 's-', color='magenta', label='Aspect Ratio')
axes[1].axhline(y=0.4, color='cyan', linestyle='--', alpha=0.5, label='Hold threshold')
axes[1].set_ylabel('Shape Metric')
axes[1].set_title('Circularity & Aspect Ratio')
axes[1].legend()

# New vs disappeared
new_counts = [tr['new_count'] for tr in touch_results]
gone_counts = [tr['gone_count'] for tr in touch_results]
axes[2].bar(frames_x, new_counts, color='green', alpha=0.7, label='New')
axes[2].bar(frames_x, [-g for g in gone_counts], color='red', alpha=0.5, label='Gone')
axes[2].set_ylabel('Pixel Count')
axes[2].set_title('New vs Disappeared Trace Pixels')
axes[2].legend()
axes[2].axhline(y=0, color='black', linewidth=0.5)

# Touch type timeline
type_nums = {'none': 0, 'residual': 1, 'fading': 2, 'hold': 3, 'swipe': 4, 'new_swipe': 5}
type_values = [type_nums.get(tr['touch_type'], 0) for tr in touch_results]
type_cols = [type_colors.get(tr['touch_type'], 'gray') for tr in touch_results]
axes[3].bar(frames_x, type_values, color=type_cols, alpha=0.8)
axes[3].set_yticks(list(type_nums.values()))
axes[3].set_yticklabels(list(type_nums.keys()))
axes[3].set_xlabel('Frame')
axes[3].set_title('Touch Type Classification')

plt.tight_layout()
plt.savefig('/home/claude/b3_metrics.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ D: Key frames zoomed - the hold pattern (frames 1-7) ---
fig, axes = plt.subplots(2, 7, figsize=(28, 8))
fig.suptitle('Hold Pattern Analysis: Frames 1-7', fontsize=14, fontweight='bold')

for col in range(7):
    fi = col
    axes[0][col].imshow(frames_rgb[fi])
    axes[0][col].set_title(f'Frame {fi+1}', fontsize=10)
    axes[0][col].axis('off')
    
    # Zoomed trace region
    mask = trace_masks[fi]
    if np.sum(mask > 0) > 20:
        coords = np.argwhere(mask > 0)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        pad = 40
        y_min = max(0, y_min - pad)
        y_max = min(h, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(w, x_max + pad)
        
        crop = frames_rgb[fi][y_min:y_max, x_min:x_max].copy()
        mask_crop = mask[y_min:y_max, x_min:x_max]
        
        # Overlay trace boundary
        contours, _ = cv2.findContours(mask_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, (0, 255, 0), 2)
        
        axes[1][col].imshow(crop)
        area = np.sum(mask_crop > 0)
        axes[1][col].set_title(f'Trace: {area}px', fontsize=9)
    else:
        axes[1][col].imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        axes[1][col].set_title('No trace', fontsize=9)
    axes[1][col].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/b3_hold_zoom.png', dpi=150, bbox_inches='tight')
plt.close()

# --- VIZ E: Transition frames 17-20 (swipe appears) ---
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('Swipe Emergence: Frames 17-20', fontsize=14, fontweight='bold')

for col, fi in enumerate(range(16, 20)):
    axes[0][col].imshow(frames_rgb[fi])
    axes[0][col].set_title(f'Frame {fi+1}', fontsize=10)
    axes[0][col].axis('off')
    
    # New trace (cyan) vs persistent (green) vs disappeared (red)
    if fi > 0:
        composite = np.zeros((h, w, 3), dtype=np.uint8)
        cur = trace_masks[fi]
        prev = trace_masks[fi-1]
        existing = cv2.bitwise_and(cur, prev)
        new_only = cv2.bitwise_and(cur, cv2.bitwise_not(prev))
        gone = cv2.bitwise_and(prev, cv2.bitwise_not(cur))
        
        composite[:, :, 1] = existing
        composite[:, :, 0] = new_only  # Red = new
        composite[:, :, 2] = gone  # Blue = disappeared
        
        axes[1][col].imshow(composite)
        axes[1][col].set_title(f'R=new({np.sum(new_only>0)}) G=persist B=gone({np.sum(gone>0)})', fontsize=8)
    axes[1][col].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/b3_swipe_emerge.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nAll batch 3 visualizations saved!")

# Summary
print("\n=== BATCH 3 SUMMARY ===")
for tr in touch_results:
    icon = {'hold': '●', 'swipe': '→', 'new_swipe': '★', 'fading': '↓', 'residual': '·', 'none': '✕'}
    print(f"  F{tr['frame']:2d}: {icon.get(tr['touch_type'],'?')} {tr['touch_type']:10s} "
          f"area={tr['cur_area']:5d} circ={tr['main_circularity']:.2f}")
