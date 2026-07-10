#!/bin/bash
# offload_corpus_to_modal.sh — upload the SLS corpus to the Modal volume, verify,
# THEN delete the local copy to free the rig. HYBRID + hardened against hangs.
#
# WHY HYBRID: `modal volume put` on a folder does one HTTP put per file. Over a
# many-thousand-file session that per-file overhead caps throughput at ~1.7 MB/s
# AND a single stalled connection wedges forever (no client-side timeout — this
# actually happened: a 29k-frame session hung 15h with a dead socket). So:
#   * small/medium sessions  -> per-frame upload, count-match + pixel spot-check.
#   * monster sessions (>= FRAME_TAR_THRESHOLD frames) -> tar the dir into ONE
#     file and upload that (kills per-file overhead + stall risk). Verified by
#     tar-contents count + pixel-fidelity spot-check from the tar + remote-size
#     match. Sessions are processed SMALLEST-FIRST so freed space is available to
#     stage a monster's tar (needs ~dir-size free) by the time we reach it.
#
# EVERY upload runs under a STALL-WATCHDOG: if the put child goes STALL_SECS with
# no CPU progress AND no open inet socket, it is killed and retried (RETRIES x).
# macOS has no `timeout`; that missing guard is exactly why the first run hung.
#
# PARANOID by design (this deletes weeks-of-collection data): a session's local
# frames are removed ONLY after its verification passes. Any failure => KEEP local,
# log, continue. Resumable + idempotent: already-uploaded sessions are re-verified
# and skip re-upload; already-deleted sessions are simply absent.
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
SPOT="${SPOT:-4}"                         # random frames to pixel-verify per session
FRAME_TAR_THRESHOLD="${FRAME_TAR_THRESHOLD:-150000}"  # >= this many frames -> tar path
STALL_SECS="${STALL_SECS:-900}"           # kill a put after this long with no progress
POLL="${POLL:-60}"                        # watchdog sample interval (s)
CPU_MIN_DELTA="${CPU_MIN_DELTA:-1.0}"     # CPU-secs/poll below which (and no socket) = stalled
RETRIES="${RETRIES:-3}"                   # put attempts before giving up on a session

log(){ echo "$(date '+%F %T') $*"; }

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

# size (bytes) of a single remote file, or -1 if absent / no size attr.
remote_size(){ "$PY" -c "
import modal, sys, os
try:
    vol = modal.Volume.from_name(sys.argv[1]); p = sys.argv[2]
    d = os.path.dirname(p) or '/'; name = os.path.basename(p)
    for e in vol.listdir(d):
        if e.path.rstrip('/').endswith(name):
            print(getattr(e, 'size', -1)); break
    else:
        print(-1)
except Exception:
    print(-1)
" "$VOL" "$1" 2>/dev/null; }

# run `modal volume put SRC DST` under a stall-watchdog with retries. Returns 0 on
# a clean (non-stalled) exit-0, else 1 after RETRIES. SRC may be a dir or a file.
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
      # alive if it burned CPU this window OR still holds an inet socket
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

# --- per-frame path: upload dir, count-match, pixel spot-check N frames, delete ---
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

  OK=$("$PY" - "$S" "$SESS" "$VOL" "$SPOT" "$MODAL" <<'PYEOF'
import sys, glob, os, random, subprocess, cv2, numpy as np
S, SESS, VOL, SPOT, MODAL = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
frames = sorted(glob.glob(os.path.join(S, '**', 'frame_*.png'), recursive=True))
random.seed(0); ok = True
for f in random.sample(frames, min(SPOT, len(frames))):
    rel = os.path.relpath(f, S)
    dl = '/tmp/_verify_' + rel.replace('/', '_')
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

# --- tar path: stage dir->one .tar, verify contents+fidelity, upload, size-match, delete ---
offload_tar(){
  local S="$1" SESS="$2" LOCAL="$3" tarf need_kb free_kb tcount lsize rsize OK
  tarf="$ROOT/$SESS.tar"
  need_kb=$(du -sk "$S" | awk '{print $1}')
  free_kb=$(df -k / | tail -1 | awk '{print $4}')
  if [ "$free_kb" -lt $((need_kb + 5000000)) ]; then   # need dir-size + ~5GB buffer
    log "  SKIP-TAR $SESS: free ${free_kb}KB < need ${need_kb}KB+buffer -> KEEP local"; return 1
  fi
  log "TAR $SESS ($LOCAL frames) -> $tarf ..."
  rm -f "$tarf"
  tar -C "$ROOT" -cf "$tarf" "$SESS" || { log "  ERROR: tar build failed $SESS -> KEEP"; rm -f "$tarf"; return 1; }
  tcount=$(tar -tf "$tarf" | grep -c 'frame_.*\.png')
  [ "$tcount" -ne "$LOCAL" ] && { log "  ERROR: tar count $tcount != $LOCAL $SESS -> KEEP"; rm -f "$tarf"; return 1; }

  # pixel-fidelity: extract N random frames FROM the tar, compare to the live dir.
  OK=$("$PY" - "$S" "$tarf" "$SPOT" <<'PYEOF'
import sys, glob, os, random, tarfile, tempfile, cv2, numpy as np
S, TARF, SPOT = sys.argv[1], sys.argv[2], int(sys.argv[3])
frames = sorted(glob.glob(os.path.join(S, '**', 'frame_*.png'), recursive=True))
sess = os.path.basename(S.rstrip('/'))                 # tar was built with `tar -C ROOT SESS`
random.seed(0); ok = True
with tarfile.open(TARF) as tf:
    for f in random.sample(frames, min(SPOT, len(frames))):
        arc = sess + '/' + os.path.relpath(f, S)       # SESS/park/.../frame.png member name
        try:
            m = tf.getmember(arc); ex = tf.extractfile(m).read()
        except KeyError:
            ok = False; continue
        b = cv2.imdecode(np.frombuffer(ex, np.uint8), cv2.IMREAD_COLOR)
        a = cv2.imread(f)
        md = int(np.abs(a.astype(int) - b.astype(int)).max()) if (a is not None and b is not None and a.shape == b.shape) else -1
        if md != 0: ok = False
print('OK' if ok else 'FAIL')
PYEOF
)
  [ "$OK" != "OK" ] && { log "  ERROR: tar fidelity spot-check FAILED $SESS -> KEEP"; rm -f "$tarf"; return 1; }

  log "  uploading $SESS.tar ($(du -h "$tarf" | awk '{print $1}'))..."
  run_put "$tarf" "/$SESS.tar" || { log "  ERROR: tar put gave up $SESS -> KEEP (tar left)"; return 1; }
  lsize=$(stat -f%z "$tarf"); rsize=$(remote_size "/$SESS.tar")
  [ "$lsize" != "$rsize" ] && { log "  ERROR: tar remote size $rsize != local $lsize $SESS -> KEEP"; return 1; }
  rm -rf "$S" "$tarf"
  log "  OK(tar): $SESS as $SESS.tar ($LOCAL frames, ${lsize}B), DELETED local (free now $(df -g / | tail -1 | awk '{print $4}')GB)"
}

# order sessions ascending by frame count (quick wins first; frees disk to stage
# a monster's tar by the time we reach it). bash 3.2 compatible (temp file loop).
log "sizing sessions (tar threshold=${FRAME_TAR_THRESHOLD} frames)..."
ORDER_FILE=$(mktemp)
for S in "$ROOT"/*/; do
  n=$(find "$S" -name 'frame_*.png' 2>/dev/null | wc -l | tr -d ' '); echo "$n $S"
done | sort -n > "$ORDER_FILE"

while IFS= read -r LINE <&3; do
  LOCAL=${LINE%% *}; S=${LINE#* }
  SESS=$(basename "$S")
  [ "$LOCAL" -eq 0 ] && { log "SKIP $SESS (0 frames)"; continue; }
  if [ "$LOCAL" -ge "$FRAME_TAR_THRESHOLD" ]; then
    offload_tar "$S" "$SESS" "$LOCAL"
  else
    offload_per_frame "$S" "$SESS" "$LOCAL"
  fi
done 3< "$ORDER_FILE"
rm -f "$ORDER_FILE"
log "OFFLOAD COMPLETE"
