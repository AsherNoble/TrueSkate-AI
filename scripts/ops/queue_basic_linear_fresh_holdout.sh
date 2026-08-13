#!/bin/bash
# Queue the stronger pooled MVP-2 run after a fresh-corpus finalizer succeeds.
# The finalizer owns collector shutdown, menu scanning and the fresh-only run;
# this wrapper deliberately waits for that process to exit before uploading the
# two isolated source subtrees to the pooled fresh-held-out volume.
set -eu

FINALIZER_PID_FILE="${1:?usage: queue_basic_linear_fresh_holdout.sh FINALIZER_PID_FILE LEGACY_OUT FRESH_OUT RUN_LABEL [target]}"
LEGACY_OUT="${2:?usage: queue_basic_linear_fresh_holdout.sh FINALIZER_PID_FILE LEGACY_OUT FRESH_OUT RUN_LABEL [target]}"
FRESH_OUT="${3:?usage: queue_basic_linear_fresh_holdout.sh FINALIZER_PID_FILE LEGACY_OUT FRESH_OUT RUN_LABEL [target]}"
RUN_LABEL="${4:?usage: queue_basic_linear_fresh_holdout.sh FINALIZER_PID_FILE LEGACY_OUT FRESH_OUT RUN_LABEL [target]}"
TARGET="${5:-1000}"
REPO=/Users/training-server/trueskate-ai

cd "$REPO"
[ -s "$FINALIZER_PID_FILE" ] || { echo "missing finalizer PID file: $FINALIZER_PID_FILE" >&2; exit 2; }
finalizer_pid="$(tr -d '[:space:]' < "$FINALIZER_PID_FILE")"
case "$finalizer_pid" in ''|*[!0-9]*) echo "invalid finalizer PID" >&2; exit 2;; esac
while kill -0 "$finalizer_pid" 2>/dev/null; do
  echo "[basic-linear-pooled-queue] waiting for fresh finalizer pid=$finalizer_pid"
  sleep 60
done
if ! grep -q 'App completed' logs/basic_linear_4k_finalizer.log; then
  echo "[basic-linear-pooled-queue] fresh finalizer did not report Modal completion; refusing pooled run" >&2
  exit 3
fi
env MODAL_CORPUS_VOLUME=trueskate-mvp-linear-mixed-fresh-v1 \
  bash scripts/ops/train_basic_linear_fresh_holdout.sh "$LEGACY_OUT" "$FRESH_OUT" "$RUN_LABEL" "$TARGET"
