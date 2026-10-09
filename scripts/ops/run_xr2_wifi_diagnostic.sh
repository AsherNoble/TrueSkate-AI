#!/bin/bash
# Live XR2 Wi-Fi diagnostic on the rig (desktop Terminal, for the admin prompt and Enter).
#   run_xr2_wifi_diagnostic.sh <run-name> [coordinator flags...]
# Pauses com.trueskate.services (its USB retry loop restarts Appium on 4726), runs a plain
# Appium on 4726 and the coordinator, then always stops Appium and restores the services
# agent. Operator prompts (Terminal, speech, ntfy) come from the coordinator's --operator.
set -u
RUN=${1:?run name required}; shift
STAGE=$(cd "$(dirname "$0")" && pwd)
REPO=/Users/training-server/trueskate-ai
OUT=$STAGE/$RUN
PLIST=/Users/training-server/Library/LaunchAgents/com.trueskate.services.plist
LOG=$STAGE/$RUN-wrapper.log
test ! -e "$OUT" || { echo "output $OUT exists"; exit 1; }
exec > >(tee -a "$LOG") 2>&1

APPIUM_PID=""
# Spoken prompts need sound; the rig is often muted. Restore the operator's setting on exit.
VOLUME=$(osascript -e 'output volume of (get volume settings)' 2>/dev/null)
MUTED=$(osascript -e 'output muted of (get volume settings)' 2>/dev/null)
restore() {
  if [ -n "$APPIUM_PID" ]; then kill "$APPIUM_PID" 2>/dev/null; wait "$APPIUM_PID" 2>/dev/null; fi
  if [ -n "$VOLUME" ]; then osascript -e "set volume output volume $VOLUME" >/dev/null 2>&1; fi
  if [ "$MUTED" = true ]; then osascript -e 'set volume with output muted' >/dev/null 2>&1; fi
  launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>&1 || true
  launchctl print "gui/$(id -u)/com.trueskate.services" | grep -E "^\s*state" | head -1
  echo "services restored $(date)"
}
trap restore EXIT
osascript -e 'set volume output volume 70 without output muted' >/dev/null 2>&1 \
  && echo "sound on for prompts (was volume ${VOLUME:-?}, muted ${MUTED:-?})" \
  || echo "WARNING: could not unmute; spoken prompts may be silent"
say "Audio check" &

launchctl bootout "gui/$(id -u)/com.trueskate.services" 2>&1 && echo "services paused $(date)"
sleep 2
if lsof -nP -iTCP:4726 -sTCP:LISTEN >/dev/null; then echo "port 4726 still held; aborting"; exit 1; fi
appium --port 4726 --allow-insecure xcuitest:xctest_screen_record > "$STAGE/$RUN-appium.log" 2>&1 &
APPIUM_PID=$!
for _ in $(seq 30); do curl -s -m2 http://127.0.0.1:4726/status >/dev/null && break; sleep 1; done
curl -s -m2 http://127.0.0.1:4726/status >/dev/null || { echo "Appium did not start"; exit 1; }
echo "appium ready pid $APPIUM_PID"

"$REPO/.venv/bin/python" "$STAGE/validate_ios_ipv4_recording.py" \
  --repo "$REPO" --env-file "$REPO/.env" --out-dir "$OUT" --admin-prompt --operator \
  --tunnel-python /Users/training-server/trueskate-ai-runtime/tmp/pmd3-xr2-20261007/venv313/bin/python "$@"
RESULT=$?
echo "coordinator exit $RESULT"
exit "$RESULT"
