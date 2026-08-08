#!/bin/bash
# One-night supervisor for the additive stationary-hold experiment.
# It deliberately never widens the task beyond one calibrated, stationary,
# non-spin hold.  It stops at 08:00 Australia/Sydney even if no model passes.
set -euo pipefail

REPO="${REPO:-/Users/training-server/trueskate-ai}"
ROOT="${ROOT:-data/basic_hold_xctest}"
TARGET="${TARGET:-1000}"
POLL_S="${POLL_S:-180}"
PY="$REPO/.venv/bin/python"
MODAL="$REPO/.venv/bin/modal"
VOLUME="${MODAL_VOLUME:-trueskate-corpus}"
MODELS="${MODELS_VOLUME:-trueskate-models}"
LOG="$REPO/logs/overnight_basic_hold_supervisor.log"
LOCK="$REPO/tmp/overnight_basic_hold_supervisor.lock"
cd "$REPO"
mkdir -p tmp logs
if ! mkdir "$LOCK" 2>/dev/null; then echo "already running"; exit 0; fi
trap 'rmdir "$LOCK"' EXIT
log(){ echo "$(TZ=Australia/Sydney date '+%F %T %Z') $*" | tee -a "$LOG"; }
notify(){ PYTHONPATH="$REPO/src" "$PY" -c "from trueskate_ai.utils.notify import notify; notify('''$1''', title='TrueSkate basic hold overnight')" 2>/dev/null || true; }
deadline_epoch="$($PY -c 'from datetime import datetime; from zoneinfo import ZoneInfo; now=datetime.now(ZoneInfo("Australia/Sydney")); end=now.replace(hour=8,minute=0,second=0,microsecond=0); print(int(end.timestamp()))')"
now_epoch(){ date +%s; }
accepted(){ PYTHONPATH=src "$PY" -c "from trueskate_ai.vision.basic_hold_dataset import BasicHoldClipDataset; print(len(BasicHoldClipDataset('$ROOT')))"; }
stop_collector(){
  local parent children pid
  parent=$(pgrep -f "mvp_collect.sh iPhone_XR $ROOT" || true)
  children=$(pgrep -f "collect_sls_xctest.py.*--basic-holds.*$ROOT" || true)
  [ -n "$parent" ] && kill -TERM $parent 2>/dev/null || true
  [ -n "$children" ] && kill -INT $children 2>/dev/null || true
  log "requested graceful collector stop parent=${parent:-none} child=${children:-none}"
}
wait_stable(){
  local end=$(( $(date +%s) + 1200 ))
  while [ "$(date +%s)" -lt "$end" ]; do
    if ! pgrep -f "collect_sls_xctest.py.*--basic-holds.*$ROOT" >/dev/null && ! find "$ROOT" -name '*.aligning' -print -quit | grep -q .; then return 0; fi
    sleep 20
  done
  return 1
}
measure_space(){ "$MODAL" run scripts/cloud/modal_volume_space.py | tail -1; }
prune_if_needed(){
  local need="$1" report="$2" available deficit candidate
  available=$(echo "$report" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["available_bytes"])')
  [ "$available" -ge "$need" ] && return 0
  deficit=$((need - available))
  log "Modal needs $deficit more bytes; pruning oldest sessions as authorized"
  while [ "$deficit" -gt 0 ]; do
    candidate=$(echo "$report" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); xs=sorted((x for x in d["sessions"] if x["bytes"]), key=lambda x:(x["mtime"],x["name"])); print((xs[0]["name"]+" "+str(xs[0]["bytes"])) if xs else "")')
    [ -n "$candidate" ] || return 1
    local name bytes; name=${candidate%% *}; bytes=${candidate##* }
    log "deleting Modal session $name ($bytes bytes)"
    "$MODAL" volume rm -r "$VOLUME" "/$name"
    report=$(measure_space)
    available=$(echo "$report" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["available_bytes"])')
    deficit=$((need - available))
  done
}
train_variant(){
  local label="$1" channels="$2" epochs="$3" session="$4" result
  log "training $label channels=$channels epochs=$epochs" >&2
  "$MODAL" run scripts/cloud/train_basic_hold_modal.py --data-subdir "$session" --run-label "$label" --base-channels "$channels" --epochs "$epochs" >> "$LOG" 2>&1
  mkdir -p tmp/overnight_basic_hold
  "$MODAL" volume get "$MODELS" "basic_hold_${label}.json" tmp/overnight_basic_hold/ >/dev/null
  result=$(cat "tmp/overnight_basic_hold/basic_hold_${label}.json")
  echo "$result" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print("pass" if d["passes_acceptance"] else "fail")'
}

while [ "$(now_epoch)" -lt "$deadline_epoch" ]; do
  n=$(accepted)
  log "accepted=$n/$TARGET"
  if [ "$n" -ge "$TARGET" ]; then
    stop_collector
    if ! wait_stable; then notify "Basic-hold collector did not settle after stop request; keeping data local."; exit 1; fi
    n=$(accepted); session=$(find "$ROOT" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | head -1)
    log "stable accepted=$n session=$session"
    report=$(measure_space)
    bytes=$(du -sk "$ROOT" | awk '{print $1 * 1024}')
    need=$((bytes * 12 / 10))
    prune_if_needed "$need" "$report" || { notify "Modal space could not be made sufficient; basic-hold data retained locally."; exit 1; }
    ROOT="$ROOT" QUIESCENT_MIN=1 bash scripts/ops/offload_corpus_to_modal.sh >> "$LOG" 2>&1
    log "upload complete; starting simple hold-only variants"
    for spec in "baseline 16 40" "wider 24 60" "capacity 32 80"; do
      [ "$(now_epoch)" -lt "$deadline_epoch" ] || break
      set -- $spec
      outcome=$(train_variant "$1" "$2" "$3" "$session")
      log "$1 outcome=$outcome"
      if [ "$outcome" = pass ]; then notify "Basic hold model passed: $1. Metrics are in trueskate-models/basic_hold_${1}.json."; exit 0; fi
    done
    notify "Basic hold variants completed without passing before 08:00 AEST; metrics are in trueskate-models."
    exit 1
  fi
  sleep "$POLL_S"
done
stop_collector
notify "08:00 AEST reached before $TARGET accepted basic-hold clips; collector stop requested."
log "deadline reached"
