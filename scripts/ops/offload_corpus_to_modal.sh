#!/bin/bash
# offload_corpus_to_modal.sh — upload the SLS corpus to the Modal volume, verify
# each session is pixel-exact on Modal, THEN delete the local copy to free the rig.
#
# PARANOID by design (this deletes weeks-of-collection data): a session's local
# frames are removed ONLY after (1) `modal volume put` succeeds, (2) the Modal-side
# frame count matches local exactly, and (3) N random frames round-trip pixel-exact.
# Any failure => the session is KEPT locally, logged, and the run continues.
#
# Resumable + safe to re-run: sessions already deleted locally are simply absent.
# Smallest sessions first, so the rig gets off a low-disk state quickly.
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
SPOT="${SPOT:-4}"                 # random frames to pixel-verify per session

log(){ echo "$(date '+%F %T') $*"; }

# order sessions ascending by frame count (quick wins + frees disk sooner).
# bash 3.2 compatible (macOS /bin/bash): temp file + while-read, no mapfile/arrays.
log "sizing sessions..."
ORDER_FILE=$(mktemp)
for S in "$ROOT"/*/; do
  n=$(find "$S" -name 'frame_*.png' 2>/dev/null | wc -l | tr -d ' '); echo "$n $S"
done | sort -n | awk '{print $2}' > "$ORDER_FILE"

while IFS= read -r S <&3; do
  SESS=$(basename "$S")
  LOCAL=$(find "$S" -name 'frame_*.png' 2>/dev/null | wc -l | tr -d ' ')
  [ "$LOCAL" -eq 0 ] && { log "SKIP $SESS (0 frames)"; continue; }

  log "UPLOAD $SESS ($LOCAL frames)..."
  if ! "$MODAL" volume put "$VOL" "$S" "/$SESS" >/dev/null 2>&1; then
    log "  ERROR: upload failed for $SESS -> KEEP local, continue"; continue
  fi

  REMOTE=$("$MODAL" volume ls -r "$VOL" "/$SESS" 2>/dev/null | grep -c 'frame_.*\.png')
  if [ "$REMOTE" -ne "$LOCAL" ]; then
    log "  ERROR: count mismatch $SESS local=$LOCAL remote=$REMOTE -> KEEP local, continue"; continue
  fi

  OK=$("$PY" - "$S" "$SESS" "$VOL" "$SPOT" "$MODAL" <<'PYEOF'
import sys, glob, os, random, subprocess, cv2, numpy as np
S, SESS, VOL, SPOT, MODAL = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
frames = sorted(glob.glob(os.path.join(S, '**', 'frame_*.png'), recursive=True))
random.seed(0)
ok = True
for f in random.sample(frames, min(SPOT, len(frames))):
    rel = os.path.relpath(f, S)                       # park/sample/frame.png
    dl = '/tmp/_verify_' + rel.replace('/', '_')
    subprocess.run([MODAL, 'volume', 'get', '--force', VOL, '/' + SESS + '/' + rel, dl],
                   capture_output=True)
    a, b = cv2.imread(f), cv2.imread(dl)
    md = int(np.abs(a.astype(int) - b.astype(int)).max()) if (a is not None and b is not None and a.shape == b.shape) else -1
    if md != 0:
        ok = False
    try: os.remove(dl)
    except OSError: pass
print('OK' if ok else 'FAIL')
PYEOF
)
  if [ "$OK" != "OK" ]; then
    log "  ERROR: pixel spot-check FAILED $SESS -> KEEP local, continue"; continue
  fi

  rm -rf "$S"
  FREE=$(df -g / | tail -1 | awk '{print $4}')
  log "  OK: verified $REMOTE frames on Modal, DELETED local $SESS (free now ${FREE}GB)"
done 3< "$ORDER_FILE"
rm -f "$ORDER_FILE"
log "OFFLOAD COMPLETE"
