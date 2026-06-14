#!/bin/bash
# Train the domain-robust (multi-park) Model 1 overnight when the laptop is idle,
# then evaluate transfer to the expert SLS-arena clips. Invoked by a transient
# caffeinate+sleep waiter (see tmp/model1_tonight_waiter.log) — NOT a persistent
# startup agent. Safe to run standalone too.
set -u
REPO="/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH=src
TS=$(date +%Y%m%d_%H%M%S)
LOG="$REPO/tmp/model1_tonight_$TS.log"
exec >> "$LOG" 2>&1
echo "=== model1 multi-park training $TS ==="

# Balanced multi-park training set (symlinks): one 250-sample warehouse session
# + every park-tagged session EXCEPT underpass (excluded per Asher). New parks
# (e.g. an SLS arena) auto-include once collected — just give them a _<park> tag.
TRAIN_DIR="$REPO/tmp/multipark_train"
rm -rf "$TRAIN_DIR"; mkdir -p "$TRAIN_DIR"
ln -s "$REPO/data/self_labeled_traces/iPhone_XR_20260614_011856" "$TRAIN_DIR/warehouse250" 2>/dev/null
i=0
for d in "$REPO"/data/self_labeled_traces/*_*; do
  [ -d "$d" ] || continue
  base="$(basename "$d")"
  case "$base" in
    *_underpass) continue ;;                       # excluded per instruction
    *[!0-9]) ;;                                     # park-tagged (ends in a letter)
    *) continue ;;                                  # bare timestamp = warehouse (already linked)
  esac
  ln -s "$d" "$TRAIN_DIR/park$i" && i=$((i+1))
done
echo "training dirs:"; ls -l "$TRAIN_DIR"

# Full-arch (416x192, base_ch 32) — laptop is idle, prioritise quality.
caffeinate -i python -u scripts/train/train_trace_extractor.py \
  --data "$TRAIN_DIR" --img-h 416 --img-w 192 --base-channels 32 \
  --epochs 25 --batch-size 16 \
  --out "$REPO/notebooks/models/trace_extractor_multipark.pth"

echo "=== transfer eval vs expert clips (on-trace% = predictions landing on a trace) ==="
python - <<'PY'
import glob, sys
import numpy as np, cv2, torch
from pathlib import Path
sys.path.insert(0, "experiments")
from gaussian_bump_predictor import GaussianBumpPredictor
def load(p):
    c = torch.load(p, map_location="cpu"); bc = c.get("base_channels", 32)
    m = GaussianBumpPredictor(3, bc); m.load_state_dict(c["model_state"]); m.eval()
    return m, c["h"], c["w"]
def warm(img, px, py, r=35):
    h, w = img.shape[:2]; x0,x1,y0,y1 = max(0,px-r),min(w,px+r),max(0,py-r),min(h,py+r)
    hsv = cv2.cvtColor(img[y0:y1,x0:x1], cv2.COLOR_BGR2HSV)
    return int(((hsv[:,:,0]<=35)&(hsv[:,:,1]>=70)&(hsv[:,:,2]>=140)).sum())
def warmtot(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return int(((hsv[:,:,0]<=35)&(hsv[:,:,1]>=70)&(hsv[:,:,2]>=140)).sum())
try:
    m, H, W = load("notebooks/models/trace_extractor_multipark.pth")
except Exception as e:
    print("eval skipped:", e); raise SystemExit(0)
clips = {"laser":"tmp/expert_frames/f_*.jpg","360dbl":"tmp/expert_test/c360/f_*.jpg","bs_dbl":"tmp/expert_test/ckick/f_*.jpg"}
for name, gl in clips.items():
    fr = sorted(glob.glob(gl))
    if not fr: continue
    warms = [warmtot(cv2.imread(f)) for f in fr]; thr = np.median(warms)+1500
    tf = [f for f,w in zip(fr,warms) if w>=thr]; on=0; confs=[]
    for f in tf:
        img = cv2.imread(f); oh,ow = img.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(img,(W,H)), cv2.COLOR_BGR2RGB).astype(np.float32)/255
        with torch.no_grad(): pr = m(torch.from_numpy(rgb).permute(2,0,1)[None])[0,0].numpy()
        py,px = np.unravel_index(int(pr.argmax()), pr.shape); confs.append(float(pr.max()))
        if warm(img, int(px/W*ow), int(py/H*oh)) > 200: on += 1
    print(f"  {name:8s} {len(tf):2d} trace frames -> on-trace {100*on/max(1,len(tf)):3.0f}%  conf {np.mean(confs):.2f}")
PY

echo "FINISHED $(date) — model at notebooks/models/trace_extractor_multipark.pth"
