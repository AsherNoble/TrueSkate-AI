#!/bin/bash
# Fleet-level ntfy watchdog for the two XCTest collectors.
#
# The installed launchd jobs still invoke this once for each XR, but each instance
# reads and atomically updates the SAME persistent fleet state.  Consequently a
# rig transition produces exactly one notification, even when both jobs observe
# it at once, and a launchd restart cannot resurrect an old incident's alerts.
#
# States:
#   healthy              both XRs are producing recent segments
#   degraded             exactly one XR is not producing
#   down                 neither XR is producing
#   pending              first boot with no recent segments; waits before paging
#
# Notifications are transition-only: an incident begins, changes severity, or
# recovers.  There are deliberately no "still down" reminders.
#
# The three positional arguments are retained for existing launchd plists but
# are not used to narrow the check: every invocation always evaluates the fleet.
set -euo pipefail

REPO="${REPO:-/Users/training-server/trueskate-ai}"
DATA="${DATA:-$REPO/data/sls_xctest}"
STATE_DIR="${TRUESKATE_WATCHDOG_STATE_DIR:-$HOME/.trueskate-watchdog}"
STATE_FILE="$STATE_DIR/collection_fleet.state"
LOCK_DIR="$STATE_DIR/collection_fleet.lock"
CHECK_INTERVAL="${CHECK_INTERVAL:-120}"
STALL_SECONDS="${STALL_SECONDS:-600}"
NEVER_ARMED_ALERT_SECONDS="${NEVER_ARMED_ALERT_SECONDS:-1800}"
LOCK_STALE_SECONDS="${LOCK_STALE_SECONDS:-600}"
WDA_STATUS_TIMEOUT="${WDA_STATUS_TIMEOUT:-4}"
WATCHDOG_ONCE="${WATCHDOG_ONCE:-0}"
PUSH_LOG="${WATCHDOG_PUSH_LOG:-}"

# Keep the existing CLI stable for its two installed launchd agents.
_legacy_device_tag="${1:-iPhone_XR2}"
_legacy_wda_port="${2:-8103}"
_legacy_label="${3:-XR2}"
readonly _legacy_device_tag _legacy_wda_port _legacy_label

LABELS=(XR1 XR2)
DEVICE_TAGS=(iPhone_XR iPhone_XR2)
WDA_PORTS=(8100 8103)

mkdir -p "$STATE_DIR"
cd "$REPO"

push() {  # $1=message $2=tag $3=priority
  if [ -n "$PUSH_LOG" ]; then
    printf '%s|%s|%s\n' "$3" "$2" "$1" >> "$PUSH_LOG"
    return
  fi
  PYTHONPATH=src .venv/bin/python -c '
import sys
from trueskate_ai.utils.notify import notify
notify(sys.argv[1], title="TrueSkate rig", tags=[sys.argv[2]],
       priority=sys.argv[3], block=True)
' "$1" "$2" "$3" 2>/dev/null \
    || echo "[fleet-watchdog] push FAILED (rc=$?) - $1"
}

newest_mtime() {  # $1=device tag
  find "$DATA" -maxdepth 2 -name 'segment_*.json' -path "*${1}_*" -type f -print0 2>/dev/null \
    | xargs -0 stat -f '%m' 2>/dev/null | sort -rn | head -1 || true
}

stack_status() {  # $1=device tag $2=WDA port
  local proc wda
  proc=$(pgrep -f "collect_sls_xctest.*--devices[= ]${1}(\$| )" >/dev/null 2>&1 && echo alive || echo DEAD)
  wda=$(curl -s -m "$WDA_STATUS_TIMEOUT" "http://localhost:${2}/status" >/dev/null 2>&1 && echo up || echo DOWN)
  echo "collector=${proc} WDA${2}=${wda}"
}

read_state() {
  PREVIOUS_STATE=""
  PREVIOUS_FAILED=""
  PREVIOUS_SINCE=0
  [ -f "$STATE_FILE" ] || return 0
  PREVIOUS_STATE=$(sed -n 's/^state=//p' "$STATE_FILE" | head -1)
  PREVIOUS_FAILED=$(sed -n 's/^failed=//p' "$STATE_FILE" | head -1)
  PREVIOUS_SINCE=$(sed -n 's/^since=//p' "$STATE_FILE" | head -1)
  case "$PREVIOUS_SINCE" in
    ''|*[!0-9]*) PREVIOUS_SINCE=0 ;;
  esac
}

write_state() {  # $1=state $2=failed labels $3=since epoch
  local tmp
  tmp=$(mktemp "$STATE_DIR/collection_fleet.XXXXXX")
  printf 'state=%s\nfailed=%s\nsince=%s\n' "$1" "$2" "$3" > "$tmp"
  mv "$tmp" "$STATE_FILE"
}

duration_text() {  # $1=seconds
  local seconds=$1 days hours minutes
  days=$((seconds / 86400))
  hours=$(((seconds % 86400) / 3600))
  minutes=$(((seconds % 3600) / 60))
  if [ "$days" -gt 0 ]; then
    printf '%dd %dh' "$days" "$hours"
  elif [ "$hours" -gt 0 ]; then
    printf '%dh %dm' "$hours" "$minutes"
  else
    printf '%dm' "$minutes"
  fi
}

snapshot_fleet() {
  local now=$1 i mtime age label tag port
  FAILED_LABELS=()
  FAILED_DETAILS=()
  for i in 0 1; do
    label=${LABELS[$i]}
    tag=${DEVICE_TAGS[$i]}
    port=${WDA_PORTS[$i]}
    mtime=$(newest_mtime "$tag")
    if [ -z "$mtime" ]; then
      age=999999
    else
      age=$((now - mtime))
    fi
    if [ "$age" -gt "$STALL_SECONDS" ]; then
      FAILED_LABELS+=("$label")
      FAILED_DETAILS+=("$label age=$((age / 60))m $(stack_status "$tag" "$port")")
    fi
  done
  if [ "${#FAILED_LABELS[@]}" -eq 0 ]; then
    FAILED_CSV=""
    FAILED_DETAIL_TEXT=""
  else
    FAILED_CSV=$(IFS=,; echo "${FAILED_LABELS[*]}")
    FAILED_DETAIL_TEXT=$(IFS='; '; echo "${FAILED_DETAILS[*]}")
  fi
  case "${#FAILED_LABELS[@]}" in
    0) CURRENT_STATE=healthy ;;
    1) CURRENT_STATE=degraded ;;
    *) CURRENT_STATE=down ;;
  esac
}

publish_transition() {  # $1=state $2=failed labels $3=details $4=since $5=now
  local state=$1 failed=$2 details=$3 since=$4 now=$5 duration
  case "$state" in
    healthy)
      duration=$(duration_text "$((now - since))")
      push "Rig recovered after ${duration}: XR1 and XR2 are producing segments again." \
        "white_check_mark" "default"
      ;;
    degraded)
      push "Rig degraded: ${failed} is not producing segments. ${details}" \
        "warning" "high"
      ;;
    down)
      push "Rig down: XR1 and XR2 are not producing segments. ${details}" \
        "warning" "high"
      ;;
  esac
}

update_fleet_state() {
  local now=$1 since
  read_state
  snapshot_fleet "$now"

  if [ "$CURRENT_STATE" = healthy ]; then
    if [ "$PREVIOUS_STATE" = down ] || [ "$PREVIOUS_STATE" = degraded ]; then
      publish_transition healthy "" "" "$PREVIOUS_SINCE" "$now"
    fi
    write_state healthy "" "$now"
    return
  fi

  # A fresh install/reboot needs a brief grace period before declaring a rig that
  # has not yet produced its first segment dead.  Persist pending so a watchdog
  # restart cannot reset the grace period indefinitely.
  if [ -z "$PREVIOUS_STATE" ]; then
    write_state pending "$FAILED_CSV" "$now"
    if [ "$NEVER_ARMED_ALERT_SECONDS" -gt 0 ]; then
      return
    fi
    PREVIOUS_STATE=pending
    PREVIOUS_SINCE=$now
  fi
  if [ "$PREVIOUS_STATE" = pending ]; then
    since=$PREVIOUS_SINCE
    if [ "$((now - since))" -lt "$NEVER_ARMED_ALERT_SECONDS" ]; then
      write_state pending "$FAILED_CSV" "$since"
      return
    fi
  elif [ "$PREVIOUS_STATE" = down ] || [ "$PREVIOUS_STATE" = degraded ]; then
    since=$PREVIOUS_SINCE
  else
    since=$now
  fi

  if [ "$PREVIOUS_STATE" = "$CURRENT_STATE" ] && [ "$PREVIOUS_FAILED" = "$FAILED_CSV" ]; then
    # The incident is unchanged.  This is intentionally the only no-op path:
    # there are no reminder notifications, regardless of outage duration.
    return
  fi

  write_state "$CURRENT_STATE" "$FAILED_CSV" "$since"
  publish_transition "$CURRENT_STATE" "$FAILED_CSV" "$FAILED_DETAIL_TEXT" "$since" "$now"
}

with_fleet_lock() {
  local now=$1 lock_mtime
  if ! mkdir "$LOCK_DIR" 2>/dev/null; then
    lock_mtime=$(stat -f '%m' "$LOCK_DIR" 2>/dev/null || echo 0)
    if [ "$((now - lock_mtime))" -gt "$LOCK_STALE_SECONDS" ]; then
      rmdir "$LOCK_DIR" 2>/dev/null || true
      mkdir "$LOCK_DIR" 2>/dev/null || return
    else
      return
    fi
  fi
  update_fleet_state "$now"
  rmdir "$LOCK_DIR" 2>/dev/null || true
}

echo "[fleet-watchdog] online: stale>${STALL_SECONDS}s, check=${CHECK_INTERVAL}s, state=$STATE_FILE"
while true; do
  now=$(date +%s)
  with_fleet_lock "$now"
  echo "$(date '+%Y-%m-%d %H:%M:%S') fleet state checked"
  [ "$WATCHDOG_ONCE" = 1 ] && break
  sleep "$CHECK_INTERVAL"
done
