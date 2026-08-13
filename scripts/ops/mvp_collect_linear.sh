#!/bin/bash
# mvp_collect_linear.sh — bounded per-segment MVP-2 finite-slope drag collector.
#
# It intentionally mirrors the healthy hold collector's calibration and persisted
# seed protocol, but writes to a separate corpus and never touches its process.
# Taps are timing controls only; the strict linear loader rejects them.
#
# Usage: scripts/ops/mvp_collect_linear.sh iPhone_XR [out_dir] [max_loops]
set -u

DEVICE="${1:?usage: mvp_collect_linear.sh DEVICE [out_dir] [max_loops]}"
OUT="${2:-data/basic_linear_xctest}"
MAX_LOOPS="${3:-1}"
REPO=/Users/training-server/trueskate-ai
PARK="The Workshop"
# The calibration gate itself remains two consistent observed taps.  A delayed
# recorder can render the first leading clapperboards before its useful window,
# so callers may increase redundant controls without weakening that gate.
CALIBRATION_TAPS_PER_SEGMENT="${BASIC_LINEAR_CALIBRATION_TAPS_PER_SEGMENT:-3}"
CALIBRATION_TAP_HOLD_S="${BASIC_LINEAR_CALIBRATION_TAP_HOLD_S:-0}"
# A shared output directory can be collected by both XRs.  Keep their seed
# streams independent; otherwise they read the same state before either writes
# it and emit identical commands, defeating command-held-out generalisation.
SEED_FILE="${BASIC_LINEAR_SEED_FILE:-$OUT/.basic_linear_next_seed_${DEVICE}}"

cd "$REPO" || exit 1
mkdir -p logs "$OUT"
if [ -s "$SEED_FILE" ] && grep -Eq '^[0-9]+$' "$SEED_FILE"; then
  next_seed=$(cat "$SEED_FILE")
else
  # Avoid a same-second collision when both XR collectors are launched together.
  # The device hash is fixed and fits the sampler's positive 31-bit seed range.
  device_seed=$(printf '%s' "$DEVICE" | cksum | awk '{print $1}')
  next_seed=$(( ( $(date +%s) + device_seed ) % 2147483647 ))
fi

i=0
while :; do
  i=$((i + 1))
  if [ "$MAX_LOOPS" -gt 0 ] && [ "$i" -gt "$MAX_LOOPS" ]; then
    echo "[mvp_collect_linear] reached max_loops=$MAX_LOOPS"; break
  fi
  seed="$next_seed"
  next_seed=$(( (next_seed + 1) % 2147483647 ))
  printf '%s\n' "$next_seed" > "$SEED_FILE"
  rejections_before=$(find "$OUT" -type f -path "*/${DEVICE}_*/*.calibration_rejected.json" | wc -l | tr -d ' ')
  echo "[mvp_collect_linear] $(date '+%H:%M:%S') segment $i on $DEVICE"
  PYTHONPATH=src .venv/bin/python scripts/data/collect_sls_xctest.py \
    --devices "$DEVICE" \
    --basic-linears \
    --tap-calibrate \
    --calibration-taps-per-segment "$CALIBRATION_TAPS_PER_SEGMENT" \
    --calibration-tap-hold-s "$CALIBRATION_TAP_HOLD_S" \
    --wait-for-align \
    --no-reset \
    --park-label "$PARK" \
    --align-video \
    --align-resize-width 128 \
    --segment-min 1 \
    --max-segments 1 \
    --seed "$seed" \
    --out-dir "$OUT" \
    --no-caffeinate
  rc=$?
  rejections_after=$(find "$OUT" -type f -path "*/${DEVICE}_*/*.calibration_rejected.json" | wc -l | tr -d ' ')
  if [ $rc -ne 0 ] && [ "$rejections_after" -gt "$rejections_before" ]; then
    echo "[mvp_collect_linear] calibration rejected — continuing with next seed"
    sleep 2
    continue
  fi
  if [ $rc -ne 0 ]; then
    echo "[mvp_collect_linear] collector exited $rc — STOPPED; recover recorder/tunnel before restart"
    break
  fi
  sleep 2
done
