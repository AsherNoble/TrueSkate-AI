#!/bin/bash
# offload_corpus_to_modal.sh — upload the SLS corpus to Modal, verify, delete local.
# CHUNKED: Modal's gateway 504s on large `modal volume put` (any put running tens of
# minutes dies with HTTP 504). So we upload each session in ~CHUNK_DIRS-sample-dir
# batches (~1GB, well under the timeout). Batches stage via cp into a per-worker
# temp dir and put to /<session>/<park>, so frames keep their real path. Sessions
# process one at a time; batches within a session run WORKERS-wide to use the uplink.
#
# Re-uploading is cheap+idempotent: Modal content-addresses blocks, so a batch that
# already landed dedups on the next attempt (no re-transfer). A session is deleted
# locally ONLY after its full remote frame count matches local. LOOP-UNTIL-DONE
# rounds ride out transient outages; every put is stall-watchdog + retry guarded.
#
# Run:  cd /Users/training-server/trueskate-ai
#       nohup bash scripts/ops/offload_corpus_to_modal.sh > logs/offload.log 2>&1 &
set -u

REPO=/Users/training-server/trueskate-ai
cd "$REPO" || exit 1
VOL="${MODAL_VOLUME:-trueskate-corpus}"
MODAL="$REPO/.venv/bin/modal"
PY="$REPO/.venv/bin/python"
ROOT=data/sls_xctest
WORKERS="${WORKERS:-3}"                     # concurrent batch uploaders
CHUNK_DIRS="${CHUNK_DIRS:-90}"              # sample dirs per batch (~11MB each -> ~1GB/put)
STALL_SECS="${STALL_SECS:-900}"
POLL="${POLL:-60}"
CPU_MIN_DELTA="${CPU_MIN_DELTA:-1.0}"
RETRIES="${RETRIES:-3}"                     # put attempts per batch PER ROUND
RETRY_BACKOFF="${RETRY_BACKOFF:-60}"
MAX_ROUNDS="${MAX_ROUNDS:-40}"
ROUND_COOLDOWN="${ROUND_COOLDOWN:-1200}"
PUT_ERR="${PUT_ERR:-logs/put_errors.log}"

CLAIMS=$(mktemp -d)
trap 'rm -rf "$CLAIMS" tmp/_stage_w* 2>/dev/null' EXIT

log(){ echo "$(date '+%F %T') [w${WID:-main}] $*"; }

cpu_secs(){
  ps -o time= -p "$1" 2>/dev/null | tr -d ' ' | awk '{
    t=$0; sub(/\..*/,"",t); n=split(t,a,":");
    if(n==3) print a[1]*3600+a[2]*60+a[3]; else if(n==2) print a[1]*60+a[2]; else print 0; }'
}

remote_count(){ "$PY" -c "
import modal, sys
try:
    vol = modal.Volume.from_name(sys.argv[1])
    print(sum(1 for e in vol.listdir(sys.argv[2], recursive=True) if e.path.endswith('.png') and 'frame_' in e.path))
except Exception:
    print(0)
" "$VOL" "$1" 2>/dev/null; }

# stall-watchdog + retry around one `modal volume put SRC DST`. Captures Modal
# stderr; on failure logs the real error (not the ╭─ Error ─╮ border) to PUT_ERR.
run_put(){
  local src="$1" dst="$2" attempt pid cpu last delta stall rc errf emsg
  attempt=1; errf=$(mktemp)
  while [ "$attempt" -le "$RETRIES" ]; do
    "$MODAL" volume put --force "$VOL" "$src" "$dst" >/dev/null 2>"$errf" &
    pid=$!; last=$(cpu_secs "$pid"); stall=0
    while kill -0 "$pid" 2>/dev/null; do
      sleep "$POLL"; kill -0 "$pid" 2>/dev/null || break
      cpu=$(cpu_secs "$pid"); delta=$(awk -v a="$cpu" -v b="$last" 'BEGIN{d=a-b;print(d<0?0:d)}'); last="$cpu"
      if awk -v d="$delta" -v m="$CPU_MIN_DELTA" 'BEGIN{exit !(d>=m)}' || lsof -nP -p "$pid" -a -i >/dev/null 2>&1; then
        stall=0
      else
        stall=$((stall + POLL))
        if [ "$stall" -ge "$STALL_SECS" ]; then
          log "  STALL ${STALL_SECS}s (try $attempt) — kill put $pid"
          pkill -KILL -P "$pid" 2>/dev/null; kill -KILL "$pid" 2>/dev/null; break
        fi
      fi
    done
    wait "$pid" 2>/dev/null; rc=$?
    if [ "$rc" -eq 0 ] && [ "$stall" -lt "$STALL_SECS" ]; then rm -f "$errf"; return 0; fi
    emsg=$(grep -iE 'status|error|exception|refused|timeout|denied|unauthor|not found|connection' "$errf" 2>/dev/null | grep -viE '^[[:space:]]*[╭╰│]' | tail -1 | tr -d '│╭╰ ')
    [ -n "$emsg" ] && echo "$(date '+%F %T') $dst try $attempt: $emsg" >> "$PUT_ERR"
    log "  put try $attempt failed (rc=$rc stall=$stall)${emsg:+: $emsg} — retry"
    attempt=$((attempt+1)); [ "$attempt" -le "$RETRIES" ] && sleep $((RETRY_BACKOFF*(attempt-1)))
  done
  rm -f "$errf"; return 1
}

# one worker: claim batch indices for the current session, cp-stage + put each.
# Reads: SESS, PARK, SAMPLE_FILE (list of sample dirs), NBATCH from the env.
batch_worker(){
  WID="$1"; local i lineno stage d
  i=0
  while [ "$i" -lt "$NBATCH" ]; do
    if mkdir "$CLAIMS/b$i" 2>/dev/null; then
      stage="tmp/_stage_w$WID"; rm -rf "$stage"; mkdir -p "$stage"
      # this batch = sample-dir lines [i*CHUNK_DIRS+1 .. (i+1)*CHUNK_DIRS]
      sed -n "$((i*CHUNK_DIRS+1)),$(((i+1)*CHUNK_DIRS))p" "$SAMPLE_FILE" | while IFS= read -r d; do
        [ -n "$d" ] && cp -R "${d%/}" "$stage/" 2>/dev/null
      done
      if run_put "$stage" "/$SESS/$PARK"; then
        log "  batch $i/$NBATCH ok"
      else
        log "  batch $i/$NBATCH FAILED (will retry next round)"
      fi
      rm -rf "$stage"
    fi
    i=$((i+1))
  done
}

# upload one session in batches, then verify + delete.
offload_session(){
  local S="$1" LOCAL PARKPATH
  SESS=$(basename "$S")
  PARKPATH=$(ls -d "$S"*/ 2>/dev/null | head -1); [ -z "$PARKPATH" ] && { log "SKIP $SESS (no park dir)"; return 1; }
  PARK=$(basename "$PARKPATH")
  SAMPLE_FILE=$(mktemp); ls -d "$PARKPATH"sample_*/ 2>/dev/null > "$SAMPLE_FILE"
  local NSAMP; NSAMP=$(grep -c . "$SAMPLE_FILE")
  NBATCH=$(( (NSAMP + CHUNK_DIRS - 1) / CHUNK_DIRS ))
  LOCAL=$(find "$S" -name 'frame_*.png' | wc -l | tr -d ' ')
  log "SESSION $SESS: $LOCAL frames, $NSAMP sample dirs -> $NBATCH batches of $CHUNK_DIRS"
  export SESS PARK SAMPLE_FILE NBATCH CHUNK_DIRS CLAIMS VOL MODAL PY PUT_ERR STALL_SECS POLL CPU_MIN_DELTA RETRIES RETRY_BACKOFF
  rm -rf "$CLAIMS"/b* 2>/dev/null
  local w; for w in $(seq 1 "$WORKERS"); do batch_worker "$w" & done
  wait
  rm -f "$SAMPLE_FILE"
  local REMOTE; REMOTE=$(remote_count "/$SESS"); REMOTE=${REMOTE:-0}
  if [ "$REMOTE" -eq "$LOCAL" ]; then
    rm -rf "$S"; log "  OK: $SESS fully on Modal ($REMOTE frames), DELETED local (free $(df -g / | tail -1 | awk '{print $4}')GB)"
  else
    log "  KEEP $SESS: remote $REMOTE != local $LOCAL (some batches pending) — retry next round"
  fi
}

local_remaining(){ local n=0 S; for S in "$ROOT"/*/; do [ -d "$S" ] && n=$((n+1)); done; echo "$n"; }

for round in $(seq 1 "$MAX_ROUNDS"); do
  [ "$(local_remaining)" -eq 0 ] && { log "nothing local — done"; break; }
  log "=== round $round/$MAX_ROUNDS: $(local_remaining) sessions local ==="
  # smallest-first so quick wins free disk sooner
  for S in $(for d in "$ROOT"/*/; do [ -d "$d" ] && echo "$(find "$d" -name 'frame_*.png' | wc -l | tr -d ' ') $d"; done | sort -n | awk '{print $2}'); do
    WID=main offload_session "$S"
  done
  [ "$(local_remaining)" -eq 0 ] && { log "all sessions offloaded"; break; }
  [ "$round" -lt "$MAX_ROUNDS" ] && { log "round $round done, $(local_remaining) remain — cooldown ${ROUND_COOLDOWN}s"; sleep "$ROUND_COOLDOWN"; }
done

if [ "$(local_remaining)" -eq 0 ]; then log "OFFLOAD COMPLETE"; else log "OFFLOAD INCOMPLETE — $(local_remaining) local after $MAX_ROUNDS rounds (see $PUT_ERR)"; fi
