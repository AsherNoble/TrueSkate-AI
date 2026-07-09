#!/bin/bash
# storage_guard.sh — halt SLS collection + alert when storage is low.
#
# The rig collapsed once from a silently-full local disk (423GB corpus filled it to
# 0GB, which broke WDA with misleading "iOS must be installed" errors). This guard
# prevents a recurrence: it checks the local SSD (and Modal volume usage, once
# Modal is set up) and, if either is low, sends an URGENT ntfy AND stops the
# collectors so we never wedge the rig by filling the disk again.
#
# Run via launchd every few minutes (scripts/ops/com.trueskate.storageguard.plist)
# on the rig, or manually. Idempotent: re-stopping already-stopped collectors is a
# no-op; it only alerts on transitions (state file) to avoid ntfy spam.
#
# Env overrides:
#   LOCAL_MIN_GB   stop collection when local free < this (default 25)
#   MODAL_MAX_GB   warn when Modal volume usage > this (default 950 of the 1024 free-tier TiB)
#   MODAL_VOLUME   Modal volume name (default trueskate-corpus)
set -u

REPO=/Users/training-server/trueskate-ai
LOCAL_MIN_GB="${LOCAL_MIN_GB:-25}"
MODAL_MAX_GB="${MODAL_MAX_GB:-950}"
MODAL_VOLUME="${MODAL_VOLUME:-trueskate-corpus}"
STATE="$HOME/.trueskate_storage_guard.state"   # last alert state, to de-dupe ntfy
LOG="$REPO/logs/storage_guard.log"
U=$(id -u)

log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG" 2>/dev/null; }
notify(){  # $1 message  $2 priority(default urgent)
  PYTHONPATH="$REPO/src" "$REPO/.venv/bin/python" -c "
from trueskate_ai.utils.notify import notify
notify('''$1''', title='TrueSkate storage guard', priority='${2:-urgent}')" 2>/dev/null || true
}
stop_collection(){ for j in collect.xr1 collect.xr2; do launchctl bootout "gui/$U/com.trueskate.$j" 2>/dev/null; done; }
set_state(){ echo "$1" > "$STATE"; }
get_state(){ cat "$STATE" 2>/dev/null || echo "OK"; }

# --- 1. LOCAL SSD (the one that killed the rig) --------------------------
FREE_GB=$(df -g / | tail -1 | awk '{print $4}')
if [ "${FREE_GB:-0}" -lt "$LOCAL_MIN_GB" ]; then
  stop_collection
  MSG="LOCAL DISK LOW: ${FREE_GB}GB free (< ${LOCAL_MIN_GB}GB). Collection STOPPED to avoid wedging the rig. Offload the corpus to Modal + free space, then restart the collectors."
  [ "$(get_state)" != "LOCAL_LOW" ] && notify "$MSG"        # alert once per transition
  set_state LOCAL_LOW
  log "LOCAL LOW ${FREE_GB}GB (<${LOCAL_MIN_GB}) -> collection stopped"
  exit 0
fi

# --- 2. MODAL volume usage (once Modal is set up) ------------------------
MODAL_MSG=""
if command -v modal >/dev/null 2>&1 && [ -f "$HOME/.modal.toml" ]; then
  # `modal volume` has no direct byte total; sum the listing. Best-effort; failures are non-fatal.
  USED_GB=$(modal volume ls "$MODAL_VOLUME" --json 2>/dev/null \
    | "$REPO/.venv/bin/python" -c "import sys,json;
try:
    d=json.load(sys.stdin); print(int(sum(f.get('size',0) for f in d)/1e9))
except Exception: print(-1)" 2>/dev/null || echo -1)
  if [ "${USED_GB:-0}" -gt "$MODAL_MAX_GB" ]; then
    MODAL_MSG="MODAL volume '$MODAL_VOLUME' near free-tier cap: ${USED_GB}GB / 1024GB. Prune old data or upgrade the plan."
    [ "$(get_state)" != "MODAL_HIGH" ] && notify "$MODAL_MSG" high
    set_state MODAL_HIGH
    log "MODAL HIGH ${USED_GB}GB"
    exit 0
  fi
fi

# --- healthy: recovery note if we were previously in an alert state -----
if [ "$(get_state)" != "OK" ]; then
  notify "Storage recovered: local ${FREE_GB}GB free. Guard clear." default
  set_state OK
fi
log "OK local=${FREE_GB}GB"
