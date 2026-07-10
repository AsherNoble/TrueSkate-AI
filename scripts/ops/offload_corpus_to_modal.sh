#!/bin/bash
# offload_corpus_to_modal.sh — upload the SLS corpus to the Modal volume, verify,
# THEN delete the local copy to free the rig. PARALLEL + stall-hardened.
#
# WHY PARALLEL: the rig's uplink is the bottleneck — ~15 Mbps capacity, but a
# single `modal volume put` connection only reaches ~0.6 MB/s (packet loss caps
# one TCP flow). WORKERS concurrent uploaders aggregate toward the ~1.88 MB/s
# ceiling (~3x). Per-frame upload for every session (no tar): the bottleneck is
# raw uplink, not per-file overhead, and per-frame keeps the full pixel-exact
# verify with zero disk staging (uploads only read; deletes only free space, so
# disk monotonically recovers even with N sessions in flight).
#
# STALL-WATCHDOG on every put: if the child goes STALL_SECS with no CPU progress
# AND no open inet socket, it is killed and retried (RETRIES x). macOS has no
# `timeout`; that missing guard is why the first run hung 15h on a dead socket.
#
# PARANOID (this deletes weeks-of-collection data): a session's local frames are
# removed ONLY after count-match on Modal AND N random frames round-trip
# pixel-exact. Any failure => KEEP local, log, continue. Idempotent/resumable:
# already-uploaded sessions skip re-upload; already-deleted are absent.
#
# Run (survives SSH drop):
#   cd /Users/training-server/trueskate-ai
#   nohup bash scripts/ops/offload_corpus_to_modal.sh > logs/offload.log 2>&1 &
set -u

REPO=/Users/training-server/trueskate-ai
cd "$REPO" || exit 1
VOL="${MODAL_VOLUME:-trueskate-corpus}"
MODAL="$REPO/.venv/bin/modal"
PY="$REPO/.venv/bin/python"
ROOT=data/sls_xctest
WORKERS="${WORKERS:-3}"                    # concurrent uploaders (target uplink ceiling)
SPOT="${SPOT:-4}"                          # random frames to pixel-verify per session
STALL_SECS="${STALL_SECS:-900}"            # kill a put after this long with no progress
POLL="${POLL:-60}"                         # watchdog sample interval (s)
CPU_MIN_DELTA="${CPU_MIN_DELTA:-1.0}"      # CPU-secs/poll below which (and no socket) = stalled
RETRIES="${RETRIES:-3}"                    # put attempts before giving up on a session

CLAIMS=$(mktemp -d)                        # atomic per-session claim dir (mkdir = lock)
ORDER_FILE=$(mktemp)
trap 'rm -rf "$CLAIMS" "$ORDER_FILE"' EXIT

log(){ echo "$(date '+%F %T') [w${WID:-main}] $*"; }

# CPU seconds consumed by a pid (ps time is [H:]MM:SS[.cc]); centiseconds dropped.
cpu_secs(){
  ps -o time= -p "$1" 2>/dev/null | tr -d ' ' | awk '{
    t=$0; sub(/\..*/,"",t); n=split(t,a,":");
    if(n==3) print a[1]*3600+a[2]*60+a[3];
    else if(n==2) print a[1]*60+a[2];
    else print 0;
  }'
}

# recursive remote frame count (modal volume ls has no -r; use the Python API).
remote_count(){ "$PY" -c "
import modal, sys
try:
    vol = modal.Volume.from_name(sys.argv[1])
    print(sum(1 for e in vol.listdir(sys.argv[2], recursive=True) if e.path.endswith('.png') and 'frame_' in e.path))
except Exception:
    print(0)
" "$VOL" "$1" 2>/dev/null; }

# run `modal volume put SRC DST` under a stall-watchdog with retries. Returns 0 on
# a clean (non-stalled) exit-0, else 1 after RETRIES.
run_put(){
  local src="$1" dst="$2" attempt pid cpu last delta stall rc
  attempt=1
  while [ "$attempt" -le "$RETRIES" ]; do
    "$MODAL" volume put --force "$VOL" "$src" "$dst" >/dev/null 2>&1 &
    pid=$!
    last=$(cpu_secs "$pid"); stall=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL"
      kill -0 "$pid" 2>/dev/null || break
      cpu=$(cpu_secs "$pid")
      delta=$(awk -v a="$cpu" -v b="$last" 'BEGIN{d=a-b; print (d<0?0:d)}')
      last="$cpu"
      if awk -v d="$delta" -v m="$CPU_MIN_DELTA" 'BEGIN{exit !(d>=m)}' \
         || lsof -nP -p "$pid" -a -i >/dev/null 2>&1; then
        stall=0
      else
        stall=$((stall + POLL))
        if [ "$stall" -ge "$STALL_SECS" ]; then
          log "  STALL: no progress ${STALL_SECS}s (attempt $attempt) — killing put $pid"
          pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null
          break
        fi
      fi
    done
    wait "$pid" 2>/dev/null; rc=$?
    if [ "$rc" -eq 0 ] && [ "$stall" -lt "$STALL_SECS" ]; then
      return 0
    fi
    log "  put attempt $attempt failed (rc=$rc, stall=$stall) — retry"
    attempt=$((attempt + 1))
  done
  return 1
}

# upload dir, count-match, pixel spot-check N frames, delete. Verify-then-delete.
offload_per_frame(){
  local S="$1" SESS="$2" LOCAL="$3" REMOTE OK
  REMOTE=$(remote_count "/$SESS"); REMOTE=${REMOTE:-0}
  if [ "$REMOTE" -ne "$LOCAL" ]; then
    log "UPLOAD $SESS ($LOCAL frames; remote has $REMOTE)..."
    run_put "$S" "/$SESS" || { log "  ERROR: put gave up $SESS -> KEEP local"; return 1; }
    REMOTE=$(remote_count "/$SESS"); REMOTE=${REMOTE:-0}
  else
    log "VERIFY $SESS (already $REMOTE frames on Modal)..."
  fi
  [ "$REMOTE" -ne "$LOCAL" ] && { log "  ERROR: count mismatch $SESS local=$LOCAL remote=$REMOTE -> KEEP"; return 1; }

  OK=$("$PY" - "$S" "$SESS" "$VOL" "$SPOT" "$MODAL" "$WID" <<'PYEOF'
import sys, glob, os, random, subprocess, cv2, numpy as np
S, SESS, VOL, SPOT, MODAL, WID = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
frames = sorted(glob.glob(os.path.join(S, '**', 'frame_*.png'), recursive=True))
random.seed(0); ok = True
for f in random.sample(frames, min(SPOT, len(frames))):
    rel = os.path.relpath(f, S)
    dl = '/tmp/_verify_w%s_%s' % (WID, rel.replace('/', '_'))   # per-worker tmp, no clobber
    subprocess.run([MODAL, 'volume', 'get', '--force', VOL, '/' + SESS + '/' + rel, dl], capture_output=True)
    a, b = cv2.imread(f), cv2.imread(dl)
    md = int(np.abs(a.astype(int) - b.astype(int)).max()) if (a is not None and b is not None and a.shape == b.shape) else -1
    if md != 0: ok = False
    try: os.remove(dl)
    except OSError: pass
print('OK' if ok else 'FAIL')
PYEOF
)
  [ "$OK" != "OK" ] && { log "  ERROR: pixel spot-check FAILED $SESS -> KEEP"; return 1; }
  rm -rf "$S"
  log "  OK: verified $REMOTE frames on Modal, DELETED local $SESS (free now $(df -g / | tail -1 | awk '{print $4}')GB)"
}

# a worker walks the size-ordered list, atomically claims each unclaimed session
# (mkdir is atomic), and offloads it. Free workers naturally grab the next job.
worker(){
  WID="$1"
  while IFS= read -r LINE; do
    local LOCAL="${LINE%% *}" S="${LINE#* }" SESS
    SESS=$(basename "$S")
    [ -d "$S" ] || continue                         # already deleted by a peer
    mkdir "$CLAIMS/$SESS" 2>/dev/null || continue   # a peer owns this session
    [ "$LOCAL" -eq 0 ] && { log "SKIP $SESS (0 frames)"; continue; }
    offload_per_frame "$S" "$SESS" "$LOCAL"
  done < "$ORDER_FILE"
}

log "sizing sessions (WORKERS=$WORKERS)..."
for S in "$ROOT"/*/; do
  n=$(find "$S" -name 'frame_*.png' 2>/dev/null | wc -l | tr -d ' '); echo "$n $S"
done | sort -n > "$ORDER_FILE"

for w in $(seq 1 "$WORKERS"); do worker "$w" & done
wait
log "OFFLOAD COMPLETE"
