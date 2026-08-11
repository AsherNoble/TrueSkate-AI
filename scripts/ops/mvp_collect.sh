#!/bin/bash
# mvp_collect.sh — per-segment collector loop for basic Model 1 holds.
#
# WHY ONE SEGMENT PER PROCESS. This is a bounded MVP pilot: every segment has its own
# manifest, a calibration report, and a contained failure surface. An earlier theory
# that a fresh process itself repaired the XCTest `started_at_epoch_s` anchor was later
# falsified, so DO NOT treat this loop as a timing fix. `--tap-calibrate` is the timing
# gate: it uses known-position rendered taps to fit each segment and preserves its .mov
# rather than generating samples when the taps disagree.
#
# GUARDS ARE OFF (--no-gameplay-guard) deliberately: the per-gesture screenshot the
# guard performs queues ahead of the next touch and added ~0.13s to onset error, and it
# cost 2.4x throughput (22 vs 52 gestures per 3 min). Single-finger stationary touches
# in a fixed park are low-risk for the editor/menu traps that motivated the guard.
# Contamination is instead caught AFTER the fact — run the flaggers over the corpus:
#     PYTHONPATH=src .venv/bin/python scripts/data/flag_editor_samples.py --root <out>
#     PYTHONPATH=src .venv/bin/python scripts/data/flag_menu_samples.py   --root <out>
#
# --no-reset matters too: reset_position is a TAP, and its own rendered mark would land
# in the next sample's window as an unlabelled touch.
#
# Usage:  scripts/ops/mvp_collect.sh iPhone_XR [out_dir] [max_loops]
set -u

DEVICE="${1:?usage: mvp_collect.sh DEVICE [out_dir] [max_loops]}"
OUT="${2:-data/mvp_xctest}"
MAX_LOOPS="${3:-1}"          # default = one go/no-go pilot; 0 = run until stopped
REPO=/Users/training-server/trueskate-ai
PARK="The Workshop"
# A second phone may append to the same corpus.  Give it its own persisted seed
# state (via BASIC_HOLD_SEED_FILE) so concurrent collectors cannot race and replay
# a random sequence; labels make the source device explicit in each sample meta.
SEED_FILE="${BASIC_HOLD_SEED_FILE:-$OUT/.basic_hold_next_seed}"

cd "$REPO" || exit 1
mkdir -p logs "$OUT"

# Each one-segment invocation creates a fresh NumPy RNG in the collector.  Leaving
# --seed at its default repeats the exact same hold/tap sequence after every
# supervisor restart, which looks like a large corpus but has almost no command
# diversity.  Persist and advance the seed *before* invoking the collector so a
# crash/restart can only skip a seed, never replay one.
if [ -s "$SEED_FILE" ] && grep -Eq '^[0-9]+$' "$SEED_FILE"; then
  next_seed=$(cat "$SEED_FILE")
else
  next_seed=$(date +%s)
fi

i=0
while :; do
  i=$((i + 1))
  if [ "$MAX_LOOPS" -gt 0 ] && [ "$i" -gt "$MAX_LOOPS" ]; then
    echo "[mvp_collect] reached max_loops=$MAX_LOOPS"; break
  fi
  seed="$next_seed"
  next_seed=$(( (next_seed + 1) % 2147483647 ))
  printf '%s\n' "$next_seed" > "$SEED_FILE"
  echo "[mvp_collect] $(date '+%H:%M:%S') segment $i on $DEVICE"
  PYTHONPATH=src .venv/bin/python scripts/data/collect_sls_xctest.py \
    --devices "$DEVICE" \
    --basic-holds \
    --tap-calibrate \
    --wait-for-align \
    --no-reset \
    --no-gameplay-guard \
    --park-label "$PARK" \
    --align-video \
    --segment-min 1 \
    --max-segments 1 \
    --seed "$seed" \
    --out-dir "$OUT" \
    --no-caffeinate
  rc=$?
  # A capped recorder-start failure is terminal. Restarting it here would hammer
  # the wedged XCTest daemon and defeat collect_sls_xctest.py's safety cap.
  if [ $rc -ne 0 ]; then
    echo "[mvp_collect] collector exited $rc — STOPPED; recover recorder/tunnel before restart"
    break
  else
    sleep 2
  fi
done
