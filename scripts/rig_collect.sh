#!/bin/bash
# Supervised XCTest SLS trace collector (one per phone). Auto-restarts on exit so a
# transient WDA/Appium hiccup doesn't end a week-long unattended run. Stop via:
#   launchctl bootout gui/$(id -u)/com.trueskate.collect.<xr1|xr2>
DEV="$1"; LABEL="$2"
cd /Users/training-server/trueskate-ai || exit 1
export TRUESKATE_MIN_FINGER_STAGGER_S=0.12  # stagger multi-finger downs -> no park-editor trigger
while true; do
  PYTHONUNBUFFERED=1 PYTHONPATH=src .venv/bin/python scripts/data/collect_sls_xctest.py \
    --devices "$DEV" --no-rotate --start-park "SLS 2015 Super Crown" \
    --segment-min 1 --capture-offset-s 0 --spin-frac 0.5
  code=$?
  PYTHONPATH=src .venv/bin/python -c "
from trueskate_ai.utils.notify import notify
notify('[$LABEL] XCTest SLS collector EXITED (code $code) — auto-restart in 30s.', title='TrueSkate rig', tags=['arrows_counterclockwise'])
" 2>/dev/null
  sleep 30
done
