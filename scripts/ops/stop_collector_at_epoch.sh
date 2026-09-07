#!/bin/bash
# Stop one bounded collector at a supplied Unix epoch without restarting rig services.
#
# The wrapper is paused before its current child receives SIGINT.  This lets the
# child save its partial segment and align it, while preventing a successor
# segment from starting during shutdown.  One completion ntfy is emitted.
set -euo pipefail

TARGET_EPOCH="${1:?usage: stop_collector_at_epoch.sh UNIX_EPOCH PID_FILE DEVICE}"
PID_FILE="${2:?usage: stop_collector_at_epoch.sh UNIX_EPOCH PID_FILE DEVICE}"
DEVICE="${3:?usage: stop_collector_at_epoch.sh UNIX_EPOCH PID_FILE DEVICE}"
REPO="${REPO:-/Users/training-server/trueskate-ai}"
LOG="${COLLECT_STOP_LOG:-$REPO/logs/scheduled_collector_stop_${DEVICE}.log}"
GRACE_SECONDS="${COLLECT_STOP_GRACE_SECONDS:-300}"

case "$TARGET_EPOCH" in ''|*[!0-9]*) echo "target epoch must be an integer" >&2; exit 2;; esac
case "$GRACE_SECONDS" in ''|*[!0-9]*) echo "grace seconds must be an integer" >&2; exit 2;; esac

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] scheduled stop armed for epoch $TARGET_EPOCH ($DEVICE)"

now=$(date +%s)
if [ "$TARGET_EPOCH" -gt "$now" ]; then
  sleep $((TARGET_EPOCH - now))
fi

if [ ! -s "$PID_FILE" ]; then
  echo "[$(date '+%F %T')] missing pid file $PID_FILE; nothing to stop"
  exit 0
fi
pid=$(tr -d '[:space:]' < "$PID_FILE")
case "$pid" in ''|*[!0-9]*) echo "[$(date '+%F %T')] invalid pid file $PID_FILE"; exit 2;; esac
if ! kill -0 "$pid" 2>/dev/null; then
  echo "[$(date '+%F %T')] collector pid $pid already stopped"
  exit 0
fi

# No new segment can start after this point; an existing child may finish cleanly.
kill -STOP "$pid"
children=$(pgrep -P "$pid" || true)
for child in $children; do
  kill -INT "$child" 2>/dev/null || true
done
echo "[$(date '+%F %T')] requested graceful stop for $DEVICE wrapper=$pid children=${children:-none}"

deadline=$(( $(date +%s) + GRACE_SECONDS ))
while :; do
  children=$(pgrep -P "$pid" || true)
  [ -z "$children" ] && break
  [ "$(date +%s)" -ge "$deadline" ] && break
  sleep 5
done

if [ -n "${children:-}" ]; then
  echo "[$(date '+%F %T')] graceful deadline reached; terminating child processes: $children"
  for child in $children; do kill -TERM "$child" 2>/dev/null || true; done
fi
kill -TERM "$pid" 2>/dev/null || true

cd "$REPO"
PYTHONPATH=src .venv/bin/python - "$DEVICE" <<'PY'
import sys
from trueskate_ai.utils.notify import notify
notify(f"[{sys.argv[1]}] scheduled collection stop completed at 09:00 AEST.",
       title="TrueSkate collection", tags=["checkered_flag"], block=True)
PY
echo "[$(date '+%F %T')] scheduled stop complete"
