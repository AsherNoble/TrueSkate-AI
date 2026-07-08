#!/bin/bash
# Laptop-side watchdog for the TrueSkate SLS rig collectors.
#
# If an XR collector stops producing recording segments (>STALL_SECONDS with no
# new segment_*.json), spawn a FRESH, BOUNDED headless Claude session to diagnose
# and recover it. The spawned session's instructions + hard permission boundaries
# live in xr_fix_agent_prompt.md (single source of truth for what it may/must-not do).
#
# WHY LAPTOP-SIDE (not a cloud cron): the fixer reaches the rig only via
# `tailscale ssh`, which needs this laptop's tailnet membership. A cloud agent
# cannot reach the rig. So this runs here, on a launchd interval — the laptop must
# be awake and on the tailnet for it to fire.
#
# The claude invocation is permission-bounded at the CLI too: --allowedTools limits
# it to `tailscale ssh ...` Bash calls only, so it literally cannot do anything on
# this laptop except talk to the rig. The prompt bounds what it does ON the rig.
#
# Schedule via scripts/ops/com.trueskate.xrwatchdog.plist, or run manually.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PROMPT_FILE="$HERE/xr_fix_agent_prompt.md"
RIG="training-server@training-server"
STALL_SECONDS="${STALL_SECONDS:-900}"          # no new segment this long => down (15 min)
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-3600}"    # don't respawn a fixer within this window
LOG="${XR_WATCHDOG_LOG:-$HOME/.claude/xr_watchdog.log}"
STAMP="$HOME/.claude/.xr_watchdog_last_spawn"
CLAUDE="${CLAUDE_BIN:-$HOME/.local/bin/claude}"
export PATH="/usr/local/bin:/opt/homebrew/bin:/Applications/Tailscale.app/Contents/MacOS:$HOME/.local/bin:$PATH"

mkdir -p "$(dirname "$LOG")"
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

# newest segment age (seconds) for a device glob; 999999 if none / unreachable
newest_age() {
  local ts
  ts=$(tailscale ssh "$RIG" "find /Users/training-server/trueskate-ai/data/sls_xctest/$1_*/ -name 'segment_*.json' -exec stat -f '%m' {} \; 2>/dev/null | sort -rn | head -1" 2>/dev/null)
  [ -z "$ts" ] && { echo 999999; return; }
  echo $(( $(date +%s) - ts ))
}

# reachability guard: if the rig/tailnet is unreachable, do nothing (not a rig fault)
if ! tailscale ssh "$RIG" 'true' 2>/dev/null; then
  log "rig unreachable (laptop off tailnet or rig down) — skipping"
  exit 0
fi

DOWN=""
for dev in iPhone_XR iPhone_XR2; do
  age=$(newest_age "$dev")
  [ "$age" -gt "$STALL_SECONDS" ] && DOWN="$DOWN $dev(${age}s)"
done

if [ -z "$DOWN" ]; then
  log "healthy — both collectors producing"
  exit 0
fi

# cooldown: avoid spawning overlapping fixers
if [ -f "$STAMP" ]; then
  last=$(cat "$STAMP" 2>/dev/null || echo 0)
  if [ $(( $(date +%s) - last )) -lt "$COOLDOWN_SECONDS" ]; then
    log "STALE:$DOWN but within cooldown — not respawning"
    exit 0
  fi
fi

date +%s > "$STAMP"
log "STALE:$DOWN — spawning bounded claude fixer"
"$CLAUDE" -p "$(cat "$PROMPT_FILE")

WATCHDOG TRIGGER: these collectors are STALE (no new segment):$DOWN. Diagnose and recover within your bounds; if the cause is out of bounds (needs sudo / disk / hands-on), stop and escalate to Asher via ntfy with the exact command." \
  --allowedTools "Bash(tailscale ssh:*)" \
  >> "$LOG" 2>&1
log "fixer session ended (exit $?)"
