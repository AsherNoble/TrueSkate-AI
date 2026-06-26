#!/bin/bash
# ntfy watchdog: alert if a device's XCTest collector STOPS producing segments.
#
# "Stopped" = no new segment_*.json written for STALL_SECONDS. This signal catches
# BOTH a dead collector AND a stuck-but-alive one (the XR1 recording-daemon-wedge
# failure mode, where the process lives but rec.start() keeps failing) — a bare
# process check would miss the latter. Alerts once on stall, re-reminds every
# REMIND_SECONDS while still down, and sends a recovery note when segments resume.
#
# Runs via launchd (com.trueskate.watchdog.<label>); KeepAlive restarts it if it dies.
# Stop:  launchctl bootout gui/$(id -u)/com.trueskate.watchdog.xr2
#
#   bash scripts/collection_watchdog.sh [DEVICE_TAG] [WDA_PORT] [LABEL]
set -u
REPO=/Users/training-server/trueskate-ai
DEVICE_TAG="${1:-iPhone_XR2}"
WDA_PORT="${2:-8103}"
LABEL="${3:-XR2}"
DATA="$REPO/data/sls_xctest"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"     # poll cadence (s)
STALL_SECONDS="${STALL_SECONDS:-600}"       # no new segment this long => alert (10 min; healthy ~75s)
REMIND_SECONDS="${REMIND_SECONDS:-3600}"    # re-remind cadence while still down (1 h)

cd "$REPO" || exit 1

push() {  # $1=message  $2=tag  $3=priority
  PYTHONPATH=src .venv/bin/python -c "
import sys
from trueskate_ai.utils.notify import notify
notify(sys.argv[1], title='TrueSkate rig - '+sys.argv[4]+' watchdog',
       tags=[sys.argv[2]], priority=sys.argv[3], block=True)
" "$1" "$2" "$3" "$LABEL" 2>/dev/null
}

# Newest segment_*.json mtime (epoch). -maxdepth 2 keeps the scan off the deep
# per-gesture sample dirs (segments live directly in the session dir).
newest_mtime() {
  find "$DATA" -maxdepth 2 -name 'segment_*.json' -path "*${DEVICE_TAG}_*" -type f -print0 2>/dev/null \
    | xargs -0 stat -f '%m' 2>/dev/null | sort -rn | head -1
}

alerted=0
last_remind=0
echo "[$LABEL-watchdog] online: tag=$DEVICE_TAG port=$WDA_PORT stall=${STALL_SECONDS}s check=${CHECK_INTERVAL}s"
push "Watchdog online - will alert if $LABEL stops collecting (no new segment > $((STALL_SECONDS/60))m)." "eyes" "low"

while true; do
  now=$(date +%s)
  mtime=$(newest_mtime)
  if [ -z "$mtime" ]; then age=999999; else age=$(( now - mtime )); fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') age=${age}s $([ "$age" -gt "$STALL_SECONDS" ] && echo STALLED || echo ok)"

  if [ "$age" -gt "$STALL_SECONDS" ]; then
    proc=$(pgrep -f "collect_sls_xctest.*${DEVICE_TAG}" >/dev/null 2>&1 && echo alive || echo DEAD)
    wda=$(curl -s -m4 "http://localhost:${WDA_PORT}/status" >/dev/null 2>&1 && echo up || echo DOWN)
    mins=$(( age / 60 ))
    if [ "$alerted" -eq 0 ]; then
      push "$LABEL STOPPED collecting - no new segment for ${mins}m. collector=${proc} WDA${WDA_PORT}=${wda}" "warning" "high"
      echo "[$LABEL-watchdog] ALERT: stalled ${mins}m (collector=$proc wda=$wda)"
      alerted=1; last_remind=$now
    elif [ $(( now - last_remind )) -ge "$REMIND_SECONDS" ]; then
      push "$LABEL STILL down - ${mins}m, no recovery. collector=${proc} WDA${WDA_PORT}=${wda}" "warning" "high"
      last_remind=$now
    fi
  else
    if [ "$alerted" -eq 1 ]; then
      push "$LABEL collecting again - recovered." "white_check_mark" "default"
      echo "[$LABEL-watchdog] RECOVERED"
      alerted=0
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
