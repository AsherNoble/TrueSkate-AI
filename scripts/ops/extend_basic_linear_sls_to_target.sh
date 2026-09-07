#!/bin/bash
# Extend the two active SLS linear corpora to a strict per-device target.
#
# Run on training-server only.  The phones must already be in their named parks;
# park labels are provenance and never navigate the game.  The script has no
# Modal dependency and emits one ntfy notification on successful completion or
# one if either collector exits before target -- never periodic reminders.
set -euo pipefail

REPO=/Users/training-server/trueskate-ai
OUT_ROOT="$REPO/data/basic_linear_sls_stage1_20260901"
TARGET="${BASIC_LINEAR_TARGET:-2000}"
XR1_OUT="$OUT_ROOT/iPhone_XR_sls_2015_super_crown"
XR2_OUT="$OUT_ROOT/iPhone_XR2_sls_2013_kansas_city"
STAMP=basic_linear_sls_2000

case "$TARGET" in ''|*[!0-9]*) echo "BASIC_LINEAR_TARGET must be an integer" >&2; exit 2;; esac
cd "$REPO"
mkdir -p "$OUT_ROOT" logs tmp

notify_once() {  # message tag
  PYTHONPATH=src .venv/bin/python - "$1" "$2" <<'PY'
import sys
from trueskate_ai.utils.notify import notify
notify(sys.argv[1], title="TrueSkate collection", priority="high", tags=[sys.argv[2]], block=True)
PY
}

strict_count() {  # root device
  PYTHONPATH=src .venv/bin/python - "$1" "$2" <<'PY'
import json
import sys
from pathlib import Path
from trueskate_ai.vision.basic_linear_dataset import discover_basic_linear_samples

root, device = Path(sys.argv[1]), sys.argv[2]
samples, _ = discover_basic_linear_samples(root)
print(sum(json.loads((sample / "meta.json").read_text()).get("device") == device for sample in samples))
PY
}

wda_healthy() { curl -fsS --max-time 5 "http://127.0.0.1:$1/status" >/dev/null; }

start_device() {  # device port park output
  local device=$1 port=$2 park=$3 out=$4 pid_file="tmp/${STAMP}_${1}.pid" watcher_file="tmp/${STAMP}_${1}_target.pid"
  local pid=""
  if ! wda_healthy "$port"; then
    notify_once "SLS collection not started: ${device} WDA :${port} is unavailable." warning
    echo "$device WDA :$port unavailable" >&2
    return 1
  fi
  if [ -s "$pid_file" ]; then pid=$(tr -d '[:space:]' < "$pid_file"); fi
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "$device collector already running pid=$pid"
  else
    BASIC_LINEAR_PARK="$park" BASIC_LINEAR_NO_MENU_GUARD=1 \
      nohup bash scripts/ops/mvp_collect_linear.sh "$device" "$out" 0 \
      >"logs/${STAMP}_${device}.log" 2>&1 &
    echo "$!" > "$pid_file"
    echo "$device collector started pid=$(cat "$pid_file")"
  fi
  if [ -s "$watcher_file" ] && kill -0 "$(tr -d '[:space:]' < "$watcher_file")" 2>/dev/null; then
    echo "$device target watcher already running"
  else
    nohup bash scripts/ops/stop_basic_linear_at_target.sh "$out" "$TARGET" "$device" "$pid_file" \
      >"logs/${STAMP}_${device}_target.log" 2>&1 &
    echo "$!" > "$watcher_file"
  fi
}

start_device iPhone_XR 8100 "SLS 2015 Super Crown" "$XR1_OUT"
start_device iPhone_XR2 8103 "SLS 2013 Kansas City" "$XR2_OUT"

# Exactly one closeout process owns completion/failure notification for this
# extension. It does not restart phones or services; a genuine failure remains
# an operator-visible incident rather than an alert loop.
nohup bash -s -- "$REPO" "$XR1_OUT" "$XR2_OUT" "$TARGET" "$STAMP" <<'FINALIZER' \
  >"logs/${STAMP}_finalize.log" 2>&1 &
set -euo pipefail
REPO=$1; XR1_OUT=$2; XR2_OUT=$3; TARGET=$4; STAMP=$5
cd "$REPO"
count() {
  PYTHONPATH=src .venv/bin/python - "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
from trueskate_ai.vision.basic_linear_dataset import discover_basic_linear_samples
samples, _ = discover_basic_linear_samples(Path(sys.argv[1]))
print(sum(json.loads((p / "meta.json").read_text()).get("device") == sys.argv[2] for p in samples))
PY
}
notice() {
  PYTHONPATH=src .venv/bin/python - "$1" "$2" <<'PY'
import sys
from trueskate_ai.utils.notify import notify
notify(sys.argv[1], title="TrueSkate collection", priority="high", tags=[sys.argv[2]], block=True)
PY
}
alive() { local f="$REPO/tmp/${STAMP}_$1.pid" p; [ -s "$f" ] || return 1; p=$(tr -d '[:space:]' < "$f"); kill -0 "$p" 2>/dev/null; }
while :; do
  xr1=$(count "$XR1_OUT" iPhone_XR); xr2=$(count "$XR2_OUT" iPhone_XR2)
  echo "[$(date '+%F %T')] XR1=$xr1 XR2=$xr2 target=$TARGET"
  if [ "$xr1" -ge "$TARGET" ] && [ "$xr2" -ge "$TARGET" ]; then
    break
  fi
  if { [ "$xr1" -lt "$TARGET" ] && ! alive iPhone_XR; } || { [ "$xr2" -lt "$TARGET" ] && ! alive iPhone_XR2; }; then
    notice "SLS collection stopped before the 2,000-clip target (XR1=${xr1}, XR2=${xr2}). Check the rig; this is the only incident alert for this run." warning
    exit 1
  fi
  sleep 60
done
for root in "$XR1_OUT" "$XR2_OUT"; do
  PYTHONPATH=src .venv/bin/python scripts/data/flag_menu_samples.py --data "$root"
done
xr1=$(count "$XR1_OUT" iPhone_XR); xr2=$(count "$XR2_OUT" iPhone_XR2)
if [ "$xr1" -lt "$TARGET" ] || [ "$xr2" -lt "$TARGET" ]; then
  notice "SLS collection reached target before menu audit but retained too few clips (XR1=${xr1}, XR2=${xr2}). This is the only incident alert for this run." warning
  exit 1
fi
notice "Hi Asher this is Codex. Switch the parks and then let me know when you've done it and which ones you switched XR1 and XR2 to." white_check_mark
FINALIZER
echo "$!" > "tmp/${STAMP}_finalize.pid"
echo "completion notifier pid=$(cat "tmp/${STAMP}_finalize.pid")"
