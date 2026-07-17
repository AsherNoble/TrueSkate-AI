#!/bin/bash
# ntfy watchdog: alert if a device's XCTest collector STOPS producing segments.
#
# "Stopped" = no new segment_*.json written for STALL_SECONDS. This signal catches
# BOTH a dead collector AND a stuck-but-alive one (the XR1 recording-daemon-wedge
# failure mode, where the process lives but rec.start() keeps failing) — a bare
# process check would miss the latter. Alerts once on stall, re-reminds every
# REMIND_SECONDS while still down, and sends a recovery note when segments resume.
# A watchdog that (re)starts while collection is ALREADY down (Mac reboot mid-
# outage) also alerts after NEVER_ARMED_ALERT_SECONDS instead of waiting silently
# forever for a first segment — a dead-from-boot rig pages too.
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
NEVER_ARMED_ALERT_SECONDS="${NEVER_ARMED_ALERT_SECONDS:-1800}"  # never saw a FIRST segment this
                                            # long after watchdog start => alert anyway (30 min)

cd "$REPO" || exit 1

push() {  # $1=message  $2=tag  $3=priority
  # A failed push must be visible in the launchd log — a watchdog whose alerts
  # silently die (broken venv, ntfy auth/network) is worse than no watchdog.
  # notify() itself bounds the send with a socket timeout, so no hang risk here.
  PYTHONPATH=src .venv/bin/python -c "
import sys
from trueskate_ai.utils.notify import notify
notify(sys.argv[1], title='TrueSkate rig - '+sys.argv[4]+' watchdog',
       tags=[sys.argv[2]], priority=sys.argv[3], block=True)
" "$1" "$2" "$3" "$LABEL" 2>/dev/null \
    || echo "[$LABEL-watchdog] push FAILED (rc=$?) - $1"
}

# Newest segment_*.json mtime (epoch). -maxdepth 2 keeps the scan off the deep
# per-gesture sample dirs (segments live directly in the session dir).
newest_mtime() {
  find "$DATA" -maxdepth 2 -name 'segment_*.json' -path "*${DEVICE_TAG}_*" -type f -print0 2>/dev/null \
    | xargs -0 stat -f '%m' 2>/dev/null | sort -rn | head -1
}

# "collector=alive/DEAD WDA<port>=up/DOWN" for alert bodies. NB the pgrep pattern
# end-anchors the device tag: a bare "iPhone_XR" would also match the iPhone_XR2
# collector's command line and report a dead XR1 collector as alive.
stack_status() {
  proc=$(pgrep -f "collect_sls_xctest.*--devices ${DEVICE_TAG}($| )" >/dev/null 2>&1 && echo alive || echo DEAD)
  wda=$(curl -s -m4 "http://localhost:${WDA_PORT}/status" >/dev/null 2>&1 && echo up || echo DOWN)
  echo "collector=${proc} WDA${WDA_PORT}=${wda}"
}

alerted=0
last_remind=0
started=$(date +%s)
seen_healthy=0   # only arm stop-alerts AFTER we've seen the device collect, so a
first=1          # benched/not-yet-started device doesn't fire a spurious "stopped".
echo "[$LABEL-watchdog] online: tag=$DEVICE_TAG port=$WDA_PORT stall=${STALL_SECONDS}s check=${CHECK_INTERVAL}s"
push "Watchdog online for $LABEL - alerts if it STOPS collecting (no new segment > $((STALL_SECONDS/60))m, once it has started)." "eyes" "low"

while true; do
  now=$(date +%s)
  mtime=$(newest_mtime)
  if [ -z "$mtime" ]; then age=999999; else age=$(( now - mtime )); fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') age=${age}s seen_healthy=${seen_healthy} $([ "$age" -gt "$STALL_SECONDS" ] && echo STALLED || echo ok)"

  if [ "$age" -le "$STALL_SECONDS" ]; then
    # --- collecting ---
    if [ "$seen_healthy" -eq 0 ]; then
      seen_healthy=1
      # Note only if we had to WAIT for it (first check already healthy => the
      # "online" message above already says it's armed) and no alert is pending
      # (then the "recovered" push below is the one that matters).
      [ "$first" -eq 0 ] && [ "$alerted" -eq 0 ] && push "$LABEL started collecting - watchdog now armed." "white_check_mark" "low"
    fi
    if [ "$alerted" -eq 1 ]; then
      push "$LABEL collecting again - recovered." "white_check_mark" "default"
      echo "[$LABEL-watchdog] RECOVERED"
      alerted=0
    fi
  else
    # --- not collecting ---
    if [ "$seen_healthy" -eq 0 ]; then
      # Never-armed hole: if the watchdog (re)starts while collection is already
      # down (Mac reboot mid-outage), waiting silently for a first segment means
      # a dead-from-boot rig NEVER pages anyone (observed 2026-07-16: ~44h silent
      # outage). Alert once the wait exceeds NEVER_ARMED_ALERT_SECONDS, then
      # re-remind on the normal cadence; the recovery push above fires when the
      # first segment finally lands.
      wait_s=$(( now - started ))
      if [ "$alerted" -eq 0 ] && [ "$wait_s" -ge "$NEVER_ARMED_ALERT_SECONDS" ]; then
        push "$LABEL has produced NO segments since watchdog start $((wait_s/60))m ago - rig may be dead from boot. $(stack_status)" "warning" "high"
        echo "[$LABEL-watchdog] ALERT: never armed after $((wait_s/60))m"
        alerted=1; last_remind=$now
      elif [ "$alerted" -eq 1 ] && [ $(( now - last_remind )) -ge "$REMIND_SECONDS" ]; then
        push "$LABEL STILL has no segments - $((wait_s/60))m since watchdog start. $(stack_status)" "warning" "high"
        last_remind=$now
      elif [ "$alerted" -eq 1 ]; then
        echo "[$LABEL-watchdog] still no first segment ($((wait_s/60))m since start; alerted)."
      else
        echo "[$LABEL-watchdog] not collecting yet (age=${age}s) - never-armed alert in $(( (NEVER_ARMED_ALERT_SECONDS - wait_s) / 60 ))m."
      fi
    else
      mins=$(( age / 60 ))
      if [ "$alerted" -eq 0 ]; then
        status=$(stack_status)
        push "$LABEL STOPPED collecting - no new segment for ${mins}m. ${status}" "warning" "high"
        echo "[$LABEL-watchdog] ALERT: stalled ${mins}m (${status})"
        alerted=1; last_remind=$now
      elif [ $(( now - last_remind )) -ge "$REMIND_SECONDS" ]; then
        push "$LABEL STILL down - ${mins}m, no recovery. $(stack_status)" "warning" "high"
        last_remind=$now
      fi
    fi
  fi
  first=0
  sleep "$CHECK_INTERVAL"
done
