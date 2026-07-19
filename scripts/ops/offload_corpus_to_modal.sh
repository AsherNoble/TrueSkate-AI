#!/bin/bash
# offload_corpus_to_modal.sh — upload the SLS corpus to Modal, verify, delete local.
# CHUNKED: Modal's gateway 504s on large `modal volume put` (any put running tens of
# minutes dies with HTTP 504). So we upload each session in ~CHUNK_DIRS-sample-dir
# batches (~1GB, well under the timeout). Batches stage via cp into a per-worker
# temp dir and put to /<session>/<park>, so frames keep their real path. Sessions
# (and the parks within a session) process one at a time; batches within a park
# run WORKERS-wide to use the uplink.
#
# Re-uploading is cheap+idempotent: Modal content-addresses blocks, so a batch that
# already landed dedups on the next attempt (no re-transfer). A session is deleted
# locally ONLY after its full remote frame count matches local. LOOP-UNTIL-DONE
# rounds ride out transient outages; every put is stall-watchdog + retry guarded.
#
# Run:  cd /Users/training-server/trueskate-ai
#       nohup bash scripts/ops/offload_corpus_to_modal.sh > logs/offload.log 2>&1 &
set -u

REPO="${REPO:-/Users/training-server/trueskate-ai}"
cd "$REPO" || exit 1
VOL="${MODAL_VOLUME:-trueskate-corpus}"
MODAL="${MODAL:-$REPO/.venv/bin/modal}"
PY="${PY:-$REPO/.venv/bin/python}"
ROOT="${ROOT:-data/sls_xctest}"
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
QUIESCENT_MIN="${QUIESCENT_MIN:-0}"         # >0: only offload sessions with NO file touched in the
                                            # last N min (never touch a session collection is still
                                            # writing). 0 = offload everything (manual one-shot).
MIN_SPIN_FRAC="${MIN_SPIN_FRAC:-0}"         # >0: require every segment manifest in a session to
                                            # record mix.spin_frac >= this value. Unknown, malformed,
                                            # mixed, and pre-spin sessions stay local. 0 = no filter.

CLAIMS=$(mktemp -d)
trap 'rm -rf "$CLAIMS" tmp/_stage_w* tmp/_stage_manifests 2>/dev/null' EXIT

log(){ echo "$(date '+%F %T') [w${WID:-main}] $*"; }

normalize_min_spin_frac(){
  "$PY" - "$1" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except (TypeError, ValueError):
    raise SystemExit(1)
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit(1)
print("0" if value == 0.0 else format(value, ".12g"))
PY
}

if ! MIN_SPIN_FRAC=$(normalize_min_spin_frac "$MIN_SPIN_FRAC"); then
  log "ERROR: MIN_SPIN_FRAC must be a finite number in [0, 1]"
  exit 2
fi

spin_filter_enabled(){ [ "$MIN_SPIN_FRAC" != "0" ]; }

# With the filter enabled, a session is eligible only when it has at least one
# segment manifest and EVERY manifest proves the requested sampler fraction.
# Fail closed on absent/malformed provenance, non-numeric values, or impossible
# fractions. The success/failure detail is suitable for the operator log.
session_has_spin_provenance(){
  local session="$1"
  spin_filter_enabled || return 0
  "$PY" - "$session" "$MIN_SPIN_FRAC" <<'PY'
import json
import math
import sys
from pathlib import Path

session = Path(sys.argv[1])
minimum = float(sys.argv[2])
manifests = sorted(session.glob("segment_*.json"))
if not manifests:
    print("no segment_*.json manifests")
    raise SystemExit(1)

values = []
for manifest in manifests:
    try:
        payload = json.loads(manifest.read_text())
        if not isinstance(payload, dict):
            raise ValueError("manifest root is not an object")
        mix = payload.get("mix")
        raw = mix.get("spin_frac") if isinstance(mix, dict) else None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError("missing/non-numeric mix.spin_frac")
        value = float(raw)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("mix.spin_frac outside [0, 1]")
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"{manifest.name}: invalid provenance ({exc})")
        raise SystemExit(1)
    if value < minimum:
        print(f"{manifest.name}: spin_frac={value:g} < {minimum:g}")
        raise SystemExit(1)
    values.append(value)

print(f"{len(manifests)} manifest(s), minimum spin_frac={min(values):g}")
PY
}

# Narrow fixture-test hook: it exercises only the provenance predicate and exits
# before any Modal lookup, upload, or local deletion.
if [ -n "${PROVENANCE_CHECK_ONLY:-}" ]; then
  session_has_spin_provenance "$PROVENANCE_CHECK_ONLY"
  exit $?
fi

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
    # strip ANSI first: rich colour-codes the border chars, defeating the ^[╭╰│] filter
    emsg=$(sed $'s/\x1b\\[[0-9;]*m//g' "$errf" 2>/dev/null | grep -iE 'status|error|exception|refused|timeout|denied|unauthor|not found|connection' | grep -viE '^[[:space:]]*[╭╰│]' | tail -1 | tr -d '│╭╰ ')
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

# upload one session in batches, then verify + delete. A session can hold MULTIPLE
# park subdirs (collect_sls_xctest.py rotates parks every --per-park-hours within
# the same session dir) — every park must be uploaded, or LOCAL (all-parks frame
# count) never matches REMOTE and the session is retried forever without progress.
offload_session(){
  local S="$1" LOCAL PARKDIRS NPARKS parkpath PROVENANCE_DETAIL
  SESS=$(basename "$S")
  if spin_filter_enabled; then
    if ! PROVENANCE_DETAIL=$(session_has_spin_provenance "$S"); then
      log "SKIP $SESS (spin provenance: $PROVENANCE_DETAIL)"
      return 0
    fi
    log "PROVENANCE $SESS: $PROVENANCE_DETAIL"
  fi
  PARKDIRS=$(ls -d "$S"*/ 2>/dev/null); [ -z "$PARKDIRS" ] && { log "SKIP $SESS (no park dir)"; return 1; }
  NPARKS=$(echo "$PARKDIRS" | grep -c .)
  LOCAL=$(find "$S" -name 'frame_*.png' | wc -l | tr -d ' ')
  log "SESSION $SESS: $LOCAL frames across $NPARKS park(s)"
  export SESS CHUNK_DIRS CLAIMS VOL MODAL PY PUT_ERR STALL_SECS POLL CPU_MIN_DELTA RETRIES RETRY_BACKOFF
  while IFS= read -r parkpath; do
    [ -z "$parkpath" ] && continue
    PARK=$(basename "$parkpath")
    SAMPLE_FILE=$(mktemp); ls -d "${parkpath}"sample_*/ 2>/dev/null > "$SAMPLE_FILE"
    local NSAMP; NSAMP=$(grep -c . "$SAMPLE_FILE")
    NBATCH=$(( (NSAMP + CHUNK_DIRS - 1) / CHUNK_DIRS ))
    log "  PARK $PARK: $NSAMP sample dirs -> $NBATCH batches of $CHUNK_DIRS"
    export PARK SAMPLE_FILE NBATCH
    rm -rf "$CLAIMS"/b* 2>/dev/null
    local w; for w in $(seq 1 "$WORKERS"); do batch_worker "$w" & done
    wait
    rm -f "$SAMPLE_FILE"
  done <<< "$PARKDIRS"
  # Session-root segment manifests (mix/device/fps/spin_frac provenance): must
  # land on the volume BEFORE the local delete or they die with it. Tiny put;
  # a failure keeps the session for the next round (dedup makes the retry free).
  local MOK=1 MSTAGE=tmp/_stage_manifests
  rm -rf "$MSTAGE"; mkdir -p "$MSTAGE"
  if cp "$S"segment_*.json "$MSTAGE"/ 2>/dev/null; then
    if run_put "$MSTAGE" "/$SESS"; then
      log "  manifests ok ($(ls "$MSTAGE" | wc -l | tr -d ' ') segment_*.json)"
    else
      MOK=0; log "  manifests FAILED — keeping session for next round"
    fi
  elif spin_filter_enabled; then
    MOK=0; log "  manifests disappeared after provenance check — keeping session"
  fi
  rm -rf "$MSTAGE"
  local REMOTE POK=1; REMOTE=$(remote_count "/$SESS"); REMOTE=${REMOTE:-0}
  # Re-check immediately before deletion so a provenance race can never turn an
  # unknown/low-spin session into a local delete after its frame upload.
  if spin_filter_enabled && ! PROVENANCE_DETAIL=$(session_has_spin_provenance "$S"); then
    POK=0; log "  provenance no longer eligible ($PROVENANCE_DETAIL) — keeping session"
  fi
  if [ "$REMOTE" -eq "$LOCAL" ] && [ "$MOK" -eq 1 ] && [ "$POK" -eq 1 ]; then
    rm -rf "$S"; log "  OK: $SESS fully on Modal ($REMOTE frames), DELETED local (free $(df -g / | tail -1 | awk '{print $4}')GB)"
  else
    log "  KEEP $SESS: remote $REMOTE vs local $LOCAL, manifests_ok=$MOK, provenance_ok=$POK — retry next round"
  fi
}

# a session is offload-eligible only if quiescent: no file touched within
# QUIESCENT_MIN minutes (cheap depth-1 check — adding a frame bumps its sample
# dir's mtime). QUIESCENT_MIN=0 => always eligible. This is what makes it SAFE to
# run while collection is live: an actively-growing session is never offloaded.
is_settled(){
  [ "$QUIESCENT_MIN" -eq 0 ] && return 0
  [ -z "$(find "$1" -maxdepth 1 -mmin -"$QUIESCENT_MIN" 2>/dev/null | head -1)" ]
}

is_offload_eligible(){
  is_settled "$1" || return 1
  session_has_spin_provenance "$1" >/dev/null
}

eligible_remaining(){ local n=0 S; for S in "$ROOT"/*/; do [ -d "$S" ] && is_offload_eligible "$S" && n=$((n+1)); done; echo "$n"; }

log_spin_provenance_skips(){
  spin_filter_enabled || return 0
  local S SESS DETAIL MARKER
  for S in "$ROOT"/*/; do
    [ -d "$S" ] && is_settled "$S" || continue
    if ! DETAIL=$(session_has_spin_provenance "$S"); then
      SESS=$(basename "$S"); MARKER="$CLAIMS/skip_$SESS"
      if mkdir "$MARKER" 2>/dev/null; then
        log "SKIP $SESS (spin provenance: $DETAIL; kept local)"
      fi
    fi
  done
}

for round in $(seq 1 "$MAX_ROUNDS"); do
  log_spin_provenance_skips
  [ "$(eligible_remaining)" -eq 0 ] && { log "nothing eligible — done"; break; }
  log "=== round $round/$MAX_ROUNDS: $(eligible_remaining) eligible sessions ==="
  # smallest-first so quick wins free disk sooner; skip non-quiescent (live) sessions
  for S in $(for d in "$ROOT"/*/; do [ -d "$d" ] && is_offload_eligible "$d" && echo "$(find "$d" -name 'frame_*.png' | wc -l | tr -d ' ') $d"; done | sort -n | awk '{print $2}'); do
    WID=main offload_session "$S"
  done
  [ "$(eligible_remaining)" -eq 0 ] && { log "all eligible sessions offloaded"; break; }
  [ "$round" -lt "$MAX_ROUNDS" ] && { log "round $round done, $(eligible_remaining) eligible remain — cooldown ${ROUND_COOLDOWN}s"; sleep "$ROUND_COOLDOWN"; }
done

if [ "$(eligible_remaining)" -eq 0 ]; then
  if spin_filter_enabled; then
    log "OFFLOAD COMPLETE — no eligible spin sessions remain; excluded sessions kept local"
  else
    log "OFFLOAD COMPLETE"
  fi
else
  log "OFFLOAD INCOMPLETE — $(eligible_remaining) eligible local after $MAX_ROUNDS rounds (see $PUT_ERR)"
fi
