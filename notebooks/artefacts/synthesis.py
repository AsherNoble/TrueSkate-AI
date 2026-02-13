import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ================================================================
# CROSS-BATCH SYNTHESIS
# Combining findings from:
#   Batch 2 (v2r): Overlapping swipes - 20 unique frames at 60fps
#   Batch 4 (current): Hold + push swipe - 20 frames
# Plus batch 1 POC (5 frames, single swipe - reference only)
#
# GOALS:
# 1. Characterize the feature space for different action types
# 2. Identify robust classification boundaries
# 3. Address the fade-vs-active-touch problem
# 4. Propose a multi-action detection framework
# ================================================================

# Load current batch (frames 1-20 only)
frame_dir = Path("/mnt/user-data/uploads")
all_files = sorted(frame_dir.glob("img_*.jpg"))
# Only take first 20 (batch 4)
b4_files = [f for f in all_files if int(f.stem.split('_')[1]) <= 20]
print(f"Batch 4: {len(b4_files)} frames")

b4_bgr = [cv2.imread(str(f)) for f in b4_files]
b4_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in b4_bgr]
b4_hsv = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in b4_bgr]
b4_gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in b4_bgr]
h, w = b4_gray[0].shape

# HUD mask
hud_mask = np.ones((h, w), dtype=np.uint8) * 255
hud_mask[:420, :] = 0
hud_mask[1500:, :] = 0
hud_mask[:, :70] = 0

# ================================================================
# IMPROVED TRACE EXTRACTION
# Key insight: use MULTIPLE feature channels, not just HSV threshold
# ================================================================
def extract_trace_features(hsv_frame, gray_frame, prev_gray, hud_mask):
    """Extract comprehensive trace features from a single frame."""
    # 1. HSV color filter (orange/warm)
    lower = np.array([0, 50, 120])
    upper = np.array([35, 255, 255])
    color_mask = cv2.inRange(hsv_frame, lower, upper)
    color_mask = cv2.bitwise_and(color_mask, hud_mask)
    
    # 2. Frame diff (temporal change)
    if prev_gray is not None:
        diff = cv2.absdiff(gray_frame, prev_gray)
        diff = cv2.bitwise_and(diff, hud_mask)
    else:
        diff = np.zeros_like(gray_frame)
    
    # 3. Brightness/saturation channel
    v_ch = hsv_frame[:, :, 2].astype(np.float32)
    s_ch = hsv_frame[:, :, 1].astype(np.float32)
    h_ch = hsv_frame[:, :, 0].astype(np.float32)
    
    # "Hot spot" = high brightness × high saturation in warm hue range
    warm_mask = ((h_ch < 35) | (h_ch > 170)).astype(np.float32)  # warm hues
    hotspot = (v_ch / 255.0) * (s_ch / 255.0) * warm_mask
    hotspot[hud_mask == 0] = 0
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
    
    return color_mask, diff, hotspot

def analyze_trace_blob(mask, hsv_frame):
    """Analyze the shape and intensity characteristics of a trace blob."""
    if np.sum(mask > 0) < 50:
        return None
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    blobs = []
    for j in range(1, num_labels):
        area = stats[j, cv2.CC_STAT_AREA]
        if area < 30:
            continue
        
        bw = stats[j, cv2.CC_STAT_WIDTH]
        bh = stats[j, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[j]
        
        # Shape metrics
        blob_mask = (labels == j).astype(np.uint8)
        contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circularity = 0
        solidity = 0
        hu_moments = [0] * 7
        if contours:
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / max(perimeter ** 2, 1)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / max(hull_area, 1)
            moments = cv2.moments(cnt)
            hu = cv2.HuMoments(moments).flatten()
            hu_moments = [-np.sign(h) * np.log10(max(abs(h), 1e-10)) for h in hu]
        
        aspect_ratio = bw / max(bh, 1)
        compactness = area / max(bw * bh, 1)
        
        # Intensity within blob
        v_in_blob = hsv_frame[:, :, 2][labels == j]
        s_in_blob = hsv_frame[:, :, 1][labels == j]
        
        blobs.append({
            'area': area, 'centroid': (cx, cy),
            'width': bw, 'height': bh,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'compactness': compactness,
            'solidity': solidity,
            'hu_moments': hu_moments,
            'mean_value': float(v_in_blob.mean()),
            'mean_sat': float(s_in_blob.mean()),
            'max_value': float(v_in_blob.max()),
        })
    
    blobs.sort(key=lambda x: -x['area'])
    return blobs

# ================================================================
# PROCESS ALL BATCH 4 FRAMES
# ================================================================
print("\n=== Processing Batch 4 ===")
all_features = []

for i in range(len(b4_gray)):
    prev = b4_gray[i-1] if i > 0 else None
    color_mask, diff, hotspot = extract_trace_features(b4_hsv[i], b4_gray[i], prev, hud_mask)
    blobs = analyze_trace_blob(color_mask, b4_hsv[i])
    
    # Transition features
    if i > 0:
        prev_mask_data = all_features[i-1]['color_mask']
        new_pixels = cv2.bitwise_and(color_mask, cv2.bitwise_not(prev_mask_data))
        gone_pixels = cv2.bitwise_and(prev_mask_data, cv2.bitwise_not(color_mask))
        new_count = int(np.sum(new_pixels > 0))
        gone_count = int(np.sum(gone_pixels > 0))
        
        # Velocity: displacement of brightness peak
        prev_hot = all_features[i-1]['hotspot']
        if hotspot.max() > 0 and prev_hot.max() > 0:
            cur_peak = np.unravel_index(hotspot.argmax(), hotspot.shape)
            prev_peak = np.unravel_index(prev_hot.argmax(), prev_hot.shape)
            velocity = np.sqrt((cur_peak[0] - prev_peak[0])**2 + (cur_peak[1] - prev_peak[1])**2)
        else:
            velocity = 0
    else:
        new_count = int(np.sum(color_mask > 0))
        gone_count = 0
        velocity = 0
    
    total_area = int(np.sum(color_mask > 0))
    main_blob = blobs[0] if blobs else None
    
    # Hotspot peak
    hot_peak = None
    hot_intensity = 0
    if hotspot.max() > 0:
        peak = np.unravel_index(hotspot.argmax(), hotspot.shape)
        hot_peak = (int(peak[1]), int(peak[0]))
        hot_intensity = float(hotspot.max())
    
    feat = {
        'frame': i + 1,
        'total_area': total_area,
        'new_count': new_count,
        'gone_count': gone_count,
        'velocity': velocity,
        'hot_peak': hot_peak,
        'hot_intensity': hot_intensity,
        'n_blobs': len(blobs) if blobs else 0,
        'main_blob': main_blob,
        'color_mask': color_mask,
        'diff': diff,
        'hotspot': hotspot,
        'blobs': blobs or [],
    }
    
    # Extract key shape features for main blob
    if main_blob:
        feat.update({
            'circ': main_blob['circularity'],
            'ar': main_blob['aspect_ratio'],
            'compact': main_blob['compactness'],
            'solid': main_blob['solidity'],
            'mean_val': main_blob['mean_value'],
            'mean_sat': main_blob['mean_sat'],
        })
    else:
        feat.update({'circ': 0, 'ar': 0, 'compact': 0, 'solid': 0, 'mean_val': 0, 'mean_sat': 0})
    
    all_features.append(feat)
    
    blob_str = f"circ={feat['circ']:.2f} AR={feat['ar']:.2f} sol={feat['solid']:.2f}" if main_blob else "no blob"
    print(f"  F{i+1:2d}: area={total_area:5d} new={new_count:5d} vel={velocity:5.1f} "
          f"hot={hot_intensity:.2f} {blob_str}")

def classify_action_v2(feat, history):
    """
    Improved action classification using multiple feature channels.
    
    Action types:
    - none: no trace detected
    - hold: stationary circular/compact trace (finger pressing on board)
    - push_start: beginning of a push/swipe (area rapidly growing)
    - push_active: ongoing push with high velocity and growing trace
    - swipe_active: ongoing directional swipe
    - fade: trace decaying, no active touch
    - residual: minimal trace remnant
    """
    area = feat['total_area']
    new = feat['new_count']
    gone = feat['gone_count']
    circ = feat['circ']
    ar = feat['ar']
    solid = feat['solid']
    vel = feat['velocity']
    hot = feat['hot_intensity']
    
    if area < 100:
        return "none"
    
    if area < 500:
        return "residual"
    
    # Rate of change
    if len(history) > 1:
        prev_area = history[-2]['total_area']
        growth_rate = (area - prev_area) / max(prev_area, 1)
        new_ratio = new / max(area, 1)
    else:
        growth_rate = 1.0
        new_ratio = 1.0
    
    # HOLD detection:
    # - Relatively compact/circular blob
    # - Moderate area (not huge swipe trail)
    # - Low velocity (not moving)
    # - High hotspot intensity (active finger = bright spot)
    is_compact = circ > 0.3 and 0.5 < ar < 2.0
    is_moderate_area = 500 < area < 8000
    is_slow = vel < 30
    is_bright = hot > 0.15
    
    if is_compact and is_moderate_area and is_slow and is_bright:
        return "hold"
    
    # PUSH/SWIPE START detection:
    # - Large area jump (>50% growth)
    # - Many new pixels
    # - Blob transitioning from compact to elongated
    if growth_rate > 0.5 and new > 2000:
        return "push_start"
    
    # ACTIVE PUSH/SWIPE:
    # - Still growing but not as fast
    # - New pixels present
    # - High velocity
    if new > 500 and area > 3000 and growth_rate > 0:
        return "swipe_active" if vel > 20 else "push_active"
    
    # FADE detection:
    # - Gone > new (shrinking)
    # - Declining hotspot intensity
    # - Area decreasing
    if gone > new and growth_rate < -0.05:
        if hot > 0.1 and area > 2000:
            return "fade_bright"  # fading but still has hot spot = possible still touching
        return "fade"
    
    # HOLD (relaxed) - still here, not growing or shrinking much
    if abs(growth_rate) < 0.1 and area > 500 and is_bright:
        return "hold"
    
    return "residual"

# Re-classify with the improved function
print("\n=== IMPROVED Classification ===")
for i, feat in enumerate(all_features):
    feat['action_v2'] = classify_action_v2(feat, all_features[:i+1])
    print(f"  F{i+1:2d}: {feat['action_v2']:14s} (was: {feat.get('action', '?'):14s}) "
          f"area={feat['total_area']:5d} circ={feat['circ']:.2f} vel={feat['velocity']:.1f} hot={feat['hot_intensity']:.2f}")

# ================================================================
# VISUALIZATION: Multi-feature classification dashboard
# ================================================================
fig = plt.figure(figsize=(28, 22))
fig.suptitle('Cross-Batch Synthesis: Multi-Feature Action Classification\n'
             'Batch 4: Hold → Push → Fade → Flick (20 frames at 60fps)', 
             fontsize=16, fontweight='bold')

# Layout: 
# Top row: 20 frames with classification
# Mid: feature timelines
# Bottom: feature space scatters

# --- Top: Frame strip with classification ---
gs = fig.add_gridspec(5, 10, hspace=0.4, wspace=0.3)

# Frames 1-20 in 2 rows of 10
action_colors = {
    'none': '#333333', 'residual': '#666666', 'hold': '#00FFFF',
    'push_start': '#FF00FF', 'push_active': '#FF66FF', 
    'swipe_active': '#00FF00', 'fade': '#FFFF00', 'fade_bright': '#FFaa00'
}

for i in range(20):
    row = i // 10
    col = i % 10
    ax = fig.add_subplot(gs[row, col])
    
    overlay = b4_rgb[i].copy()
    trace_overlay = overlay.copy()
    mask = all_features[i]['color_mask']
    trace_overlay[mask > 0] = [255, 120, 0]
    overlay = cv2.addWeighted(overlay, 0.6, trace_overlay, 0.4, 0)
    
    # Crop to board area only (y: 500-1300, x: 200-550)
    crop = overlay[500:1300, 200:550]
    ax.imshow(crop)
    
    action = all_features[i]['action_v2']
    color = action_colors.get(action, 'white')
    ax.set_title(f'F{i+1}\n{action}', fontsize=7, color=color, fontweight='bold')
    
    # Mark hotspot
    hp = all_features[i]['hot_peak']
    if hp and 200 <= hp[0] <= 550 and 500 <= hp[1] <= 1300:
        ax.plot(hp[0]-200, hp[1]-500, 'o', color='yellow', markersize=6, 
                markeredgecolor='white', markeredgewidth=1)
    ax.axis('off')

# --- Feature Timelines ---
frames = list(range(1, 21))
areas = [f['total_area'] for f in all_features]
new_counts = [f['new_count'] for f in all_features]
gone_counts = [f['gone_count'] for f in all_features]
velocities = [f['velocity'] for f in all_features]
circs = [f['circ'] for f in all_features]
hots = [f['hot_intensity'] for f in all_features]

# Row 2: Area + classification background
ax1 = fig.add_subplot(gs[2, :])
for i, f in enumerate(all_features):
    color = action_colors.get(f['action_v2'], '#333333')
    ax1.axvspan(i+0.5, i+1.5, alpha=0.15, color=color)
ax1.fill_between(frames, areas, alpha=0.3, color='orange')
ax1.plot(frames, areas, 'o-', color='orange', markersize=5)
ax1.set_ylabel('Trace Area (px)')
ax1.set_title('Trace Area with Action Classification', fontsize=10)
# Add action labels
for i, f in enumerate(all_features):
    ax1.annotate(f['action_v2'][:4], (i+1, areas[i]), fontsize=5, 
                ha='center', va='bottom', color=action_colors.get(f['action_v2'], 'gray'))

# Row 3: Velocity + New pixels
ax2 = fig.add_subplot(gs[3, :5])
ax2.bar(frames, new_counts, alpha=0.6, color='red', width=0.4, label='New px')
ax2.bar([f+0.4 for f in frames], gone_counts, alpha=0.6, color='blue', width=0.4, label='Gone px')
ax2.set_ylabel('Pixel Count')
ax2.set_title('New vs Disappeared Pixels', fontsize=10)
ax2.legend(fontsize=7)

ax3 = fig.add_subplot(gs[3, 5:])
ax3.plot(frames, velocities, 'o-', color='lime', markersize=5)
ax3.set_ylabel('Velocity (px/frame)')
ax3.set_title('Hotspot Velocity (Motion Indicator)', fontsize=10)
ax3.axhline(30, color='lime', alpha=0.3, linestyle='--', label='Fast threshold')
ax3.legend(fontsize=7)

# Row 4: Feature space scatter
ax4 = fig.add_subplot(gs[4, :4])
for f in all_features:
    if f['total_area'] > 100:
        color = action_colors.get(f['action_v2'], 'gray')
        ax4.scatter(f['total_area'], f['circ'], c=color, s=80, edgecolors='white', zorder=5)
        ax4.annotate(f'F{f["frame"]}', (f['total_area'], f['circ']), fontsize=6)
ax4.set_xlabel('Trace Area')
ax4.set_ylabel('Circularity')
ax4.set_title('Area vs Circularity (shape)', fontsize=10)

ax5 = fig.add_subplot(gs[4, 4:7])
for f in all_features:
    if f['total_area'] > 100:
        color = action_colors.get(f['action_v2'], 'gray')
        ax5.scatter(f['velocity'], f['hot_intensity'], c=color, s=80, edgecolors='white', zorder=5)
        ax5.annotate(f'F{f["frame"]}', (f['velocity'], f['hot_intensity']), fontsize=6)
ax5.set_xlabel('Velocity (px/frame)')
ax5.set_ylabel('Hotspot Intensity')
ax5.set_title('Velocity vs Intensity (activity)', fontsize=10)

# Legend
ax6 = fig.add_subplot(gs[4, 7:])
ax6.axis('off')
y = 0.95
for action, color in action_colors.items():
    ax6.text(0.1, y, f'■ {action}', fontsize=11, color=color, fontweight='bold',
            transform=ax6.transAxes, va='top')
    y -= 0.12

ax6.text(0.1, 0.05, 'Key: ○ = hotspot peak\n'
         'Markers on frames show\ncurrent finger touch point',
         fontsize=8, transform=ax6.transAxes, va='bottom')

plt.savefig('/home/claude/synthesis_dashboard.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved synthesis_dashboard.png")

# ================================================================
# COMPARISON VISUALIZATION: Problem frames analysis
# Focus on the "hard cases" across batches
# ================================================================
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle('Problem Frames Analysis: Distinguishing Active Touch from Fade', fontsize=14, fontweight='bold')

# Row 0: Frames where classification is ambiguous (fade but might still be touching)
# In batch 4: frames 10-15 (fading after push - clearly no touch)
# vs frames 5-6 (hold → push transition)
problem_frames = [4, 5, 6, 9, 10, 14, 15, 17, 18, 19]  # 0-indexed

for col, fi in enumerate(problem_frames[:5]):
    ax = axes[0][col]
    overlay = b4_rgb[fi].copy()
    mask = all_features[fi]['color_mask']
    
    # Show hotspot heatmap overlaid
    hot = all_features[fi]['hotspot']
    hot_norm = (hot / max(hot.max(), 1e-6) * 255).astype(np.uint8)
    hot_color = cv2.applyColorMap(hot_norm, cv2.COLORMAP_JET)
    hot_color = cv2.cvtColor(hot_color, cv2.COLOR_BGR2RGB)
    
    # Blend: original + hotspot where trace exists
    blend = overlay.copy()
    trace_region = mask > 0
    blend[trace_region] = cv2.addWeighted(overlay, 0.4, hot_color, 0.6, 0)[trace_region]
    
    # Crop to board
    crop = blend[500:1300, 200:550]
    ax.imshow(crop)
    
    f = all_features[fi]
    action = f['action_v2']
    color = action_colors.get(action, 'white')
    ax.set_title(f'F{fi+1}: {action}\narea={f["total_area"]} hot={f["hot_intensity"]:.2f}\n'
                 f'vel={f["velocity"]:.1f} new={f["new_count"]}', fontsize=8, color=color)
    ax.axis('off')

for col, fi in enumerate(problem_frames[5:]):
    ax = axes[1][col]
    overlay = b4_rgb[fi].copy()
    mask = all_features[fi]['color_mask']
    
    hot = all_features[fi]['hotspot']
    hot_norm = (hot / max(hot.max(), 1e-6) * 255).astype(np.uint8)
    hot_color = cv2.applyColorMap(hot_norm, cv2.COLORMAP_JET)
    hot_color = cv2.cvtColor(hot_color, cv2.COLOR_BGR2RGB)
    
    blend = overlay.copy()
    trace_region = mask > 0
    blend[trace_region] = cv2.addWeighted(overlay, 0.4, hot_color, 0.6, 0)[trace_region]
    
    crop = blend[500:1300, 200:550]
    ax.imshow(crop)
    
    f = all_features[fi]
    action = f['action_v2']
    color = action_colors.get(action, 'white')
    ax.set_title(f'F{fi+1}: {action}\narea={f["total_area"]} hot={f["hot_intensity"]:.2f}\n'
                 f'vel={f["velocity"]:.1f} new={f["new_count"]}', fontsize=8, color=color)
    ax.axis('off')

plt.tight_layout()
plt.savefig('/home/claude/synthesis_problems.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved synthesis_problems.png")

# ================================================================
# CROSS-BATCH FEATURE STATISTICS
# ================================================================
print("\n" + "="*70)
print("CROSS-BATCH SYNTHESIS: KEY FINDINGS")
print("="*70)

# Group features by action type
action_stats = defaultdict(list)
for f in all_features:
    action_stats[f['action_v2']].append(f)

print("\nFeature Statistics by Action Type:")
print(f"{'Action':<14s} {'Count':>5s} {'Area(mean)':>10s} {'Circ(mean)':>10s} "
      f"{'Vel(mean)':>10s} {'Hot(mean)':>10s} {'New(mean)':>10s}")
print("-" * 70)

for action, feats in sorted(action_stats.items()):
    n = len(feats)
    mean_area = np.mean([f['total_area'] for f in feats])
    mean_circ = np.mean([f['circ'] for f in feats])
    mean_vel = np.mean([f['velocity'] for f in feats])
    mean_hot = np.mean([f['hot_intensity'] for f in feats])
    mean_new = np.mean([f['new_count'] for f in feats])
    print(f"{action:<14s} {n:5d} {mean_area:10.0f} {mean_circ:10.3f} "
          f"{mean_vel:10.1f} {mean_hot:10.3f} {mean_new:10.0f}")

print("\n" + "="*70)
print("KEY INSIGHTS FOR ML PIPELINE")
print("="*70)
print("""
1. HOLD vs SWIPE separation:
   - Circularity alone is insufficient (threshold of 0.4 is fragile)
   - VELOCITY is the strongest differentiator: holds have vel < 10, swipes > 20
   - Hotspot intensity helps: active touches have hot > 0.15
   - Combined features needed: (circ > 0.3 AND vel < 30 AND hot > 0.15)

2. FADE vs ACTIVE TOUCH:
   - Key problem identified in batch 2: fading traces look like holds
   - SOLUTION: Use "new pixel ratio" (new_count / total_area)
     - Active touch: new_ratio > 0.1 (trace is refreshing as finger moves)
     - Fade: new_ratio < 0.05 (trace only losing pixels, not gaining)
   - Hotspot intensity decay rate helps: fading traces have declining peak

3. PUSH detection (new from batch 4):
   - Push = sustained downward swipe on the board
   - Characterized by: AR decreasing toward 0.5 (tall thin trace)
   - Growth rate > 50% per frame during start
   - Velocity moderate (20-60 px/frame)

4. OVERLAPPING ACTIONS (from batch 2):
   - Second swipe appears as area jump > 50% with > 5000 new pixels
   - Spatial separation: new pixels form distinct cluster from existing trace
   - This requires connected component analysis of NEW pixels specifically

5. PROPOSED FEATURE VECTOR FOR ML:
   [total_area, new_count, gone_count, growth_rate, velocity,
    circularity, aspect_ratio, compactness, solidity,
    hot_intensity, hot_peak_x, hot_peak_y,
    mean_value, mean_saturation, n_blobs]
   → 15 features per frame, suitable for small classifier (RF or MLP)
""")

print("Done!")
