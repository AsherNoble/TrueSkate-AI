#!/bin/bash
# mvp_collect_linear.sh — bounded per-segment MVP-2 finite-slope drag collector.
#
# It intentionally mirrors the healthy hold collector's calibration and persisted
# seed protocol, but writes to a separate corpus and never touches its process.
# Taps are timing controls only; the strict linear loader rejects them.
#
# Usage: scripts/ops/mvp_collect_linear.sh iPhone_XR [out_dir] [max_loops]
# For the device-balanced Stage 1 tranche, give every phone its own out_dir
# (e.g. data/basic_linear_stage1_20260831/iPhone_XR).  This makes provenance,
# seed state, target guards, and later audits independent by construction.
set -u

DEVICE="${1:?usage: mvp_collect_linear.sh DEVICE [out_dir] [max_loops]}"
OUT="${2:-data/basic_linear_xctest}"
MAX_LOOPS="${3:-1}"
REPO=/Users/training-server/trueskate-ai
# ``--park-label`` is provenance, not navigation: the phone must already be
# loaded in this park. Keep domain-specific collections in separate output
# roots and set BASIC_LINEAR_PARK explicitly rather than mislabelling them as
# The Workshop.
PARK="${BASIC_LINEAR_PARK:-The Workshop}"
# The calibration gate itself remains two consistent observed taps.  A delayed
# recorder can render the first leading clapperboards before its useful window,
# so callers may increase redundant controls without weakening that gate.
CALIBRATION_TAPS_PER_SEGMENT="${BASIC_LINEAR_CALIBRATION_TAPS_PER_SEGMENT:-3}"
# A 50ms ActionChains press still has ``tap`` provenance (and strict loaders
# exclude it), but is much more consistently visible to the XCTest timing
# calibrator than Appium's instantaneous mobile:tap on XR2.
CALIBRATION_TAP_HOLD_S="${BASIC_LINEAR_CALIBRATION_TAP_HOLD_S:-0.05}"
MENU_GUARD_ARGS=()
if [ "${BASIC_LINEAR_NO_MENU_GUARD:-0}" = "1" ]; then
  # SLS parks can render a persistent five-cell bottom strip that the generic
  # app-hub detector mistakes for a menu. The OS foreground guard stays ON.
  MENU_GUARD_ARGS=(--no-menu-guard)
fi
# A shared output directory can be collected by both XRs.  Keep their seed
# streams independent; otherwise they read the same state before either writes
# it and emit identical commands, defeating command-held-out generalisation.
SEED_FILE="${BASIC_LINEAR_SEED_FILE:-$OUT/.basic_linear_next_seed_${DEVICE}}"
HEARTBEAT_FILE="${BASIC_LINEAR_HEARTBEAT_FILE:-$OUT/.collector_heartbeat_${DEVICE}.json}"

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
  # ``-u`` makes every phase visible immediately in the nohup log.  Without it
  # Python's redirected stdout only flushed when the segment ended, mimicking a
  # frozen collector for the entire record/retrieve/align cycle.
  PYTHONPATH=src .venv/bin/python -u scripts/data/collect_sls_xctest.py \
    --devices "$DEVICE" \
    --basic-linears \
    --tap-calibrate \
    --calibration-taps-per-segment "$CALIBRATION_TAPS_PER_SEGMENT" \
    --calibration-tap-hold-s "$CALIBRATION_TAP_HOLD_S" \
    --wait-for-align \
    --no-reset \
    --reset-before-segment \
    "${MENU_GUARD_ARGS[@]}" \
    --park-label "$PARK" \
    --align-video \
    --align-resize-width 128 \
    --segment-min 1 \
    --max-segments 1 \
    --seed "$seed" \
    --out-dir "$OUT" \
    --heartbeat-path "$HEARTBEAT_FILE" \
    --no-run-notifications \
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
