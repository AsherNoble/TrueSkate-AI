#!/bin/bash
# Start the no-reset, calibrated Model-1 linear Stage 1 tranche on both XR phones.
#
# Run ONLY on training-server.  It starts no services, reboots no phones, and
# does not touch Modal.  Each device gets a separate corpus, persisted seed,
# collector PID, and strict-admission target watcher.
set -eu

REPO=/Users/training-server/trueskate-ai
STAMP=20260831
OUT_ROOT="/Users/training-server/trueskate-ai/data/basic_linear_stage1_${STAMP}"
TARGET=1100

cd "$REPO"
mkdir -p "$OUT_ROOT" logs tmp

start_device() {
  device="$1"
  out="$OUT_ROOT/$device"
  collector_pid="tmp/basic_linear_stage1_${device}.pid"
  watcher_pid="tmp/basic_linear_stage1_${device}_target.pid"
  mkdir -p "$out"
  if [ -s "$collector_pid" ] && kill -0 "$(tr -d '[:space:]' < "$collector_pid")" 2>/dev/null; then
    echo "$device collector already running (pid $(cat "$collector_pid"))" >&2
    return 1
  fi
  if [ -s "$watcher_pid" ] && kill -0 "$(tr -d '[:space:]' < "$watcher_pid")" 2>/dev/null; then
    echo "$device target watcher already running (pid $(cat "$watcher_pid"))" >&2
    return 1
  fi
  nohup bash scripts/ops/mvp_collect_linear.sh "$device" "$out" 0 \
    >"logs/basic_linear_stage1_${device}.log" 2>&1 &
  echo "$!" > "$collector_pid"
  nohup bash scripts/ops/stop_basic_linear_at_target.sh "$out" "$TARGET" "$device" "$collector_pid" \
    >"logs/basic_linear_stage1_${device}_target.log" 2>&1 &
  echo "$!" > "$watcher_pid"
  echo "$device started: collector=$(cat "$collector_pid") watcher=$(cat "$watcher_pid") out=$out"
}

start_device iPhone_XR
start_device iPhone_XR2

# This one-shot finalizer performs only the predeclared non-Modal close-out:
# menu flagging, audit, and one notification. It deliberately does not train.
nohup bash scripts/ops/finalize_basic_linear_stage1.sh \
  >logs/basic_linear_stage1_finalize.log 2>&1 &
echo "$!" > tmp/basic_linear_stage1_finalize.pid
echo "stage finalizer=$(cat tmp/basic_linear_stage1_finalize.pid)"
