#!/bin/bash
# Legacy launchd entrypoint for experimental mixed SLS collection.
# Fleet watchdog owns notifications; recorder start-failure caps must stay effective.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
DEVICE="${1:?usage: rig_collect.sh DEVICE [LABEL]}"
export TRUESKATE_MIN_FINGER_STAGGER_S="${TRUESKATE_MIN_FINGER_STAGGER_S:-0.12}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
# The collector already tolerates individual segment failures. Exit on its
# start-failure cap so recovery can address the recorder instead of hammering it.
exec "$REPO/.venv/bin/python" scripts/collection/collect_sls_xctest.py \
  --devices "$DEVICE" --no-rotate \
  --start-park "${TRUESKATE_SLS_PARK:-SLS 2015 Super Crown}" \
  --segment-min 1 --spin-frac "${TRUESKATE_SPIN_FRAC:-0.5}" \
  --no-run-notifications
