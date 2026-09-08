#!/bin/bash
# schedule_claude_handoff_once.sh
#
# Schedules a one-shot LaunchAgent to run a new Claude Code session in +5 hours.
# The launched session is pinned to:
#   - model:  fable
#   - effort: max
#   - context: tmp/HANDOFF_spin_quality_20260717.md
#
# It composes a prompt that tells Claude to:
#   1) use the handoff context,
#   2) inspect memory files and recent sessions,
#   3) verify whether listed tasks were actually completed,
#   4) continue unfinished branch work.
#
# Usage:
#   bash scripts/ops/schedule_claude_handoff_once.sh
#   DELAY_HOURS=5 bash scripts/ops/schedule_claude_handoff_once.sh
#   DELAY_SECONDS=18000 bash scripts/ops/schedule_claude_handoff_once.sh
#
# Verify:
#   launchctl print gui/$(id -u)/com.trueskate.claude-handoff-once | head
#   tail -f "$HOME/.claude/scheduled/com.trueskate.claude-handoff-once.run.log"
#
# Cancel:
#   launchctl bootout gui/$(id -u)/com.trueskate.claude-handoff-once 2>/dev/null || true
#   rm -f "$HOME/Library/LaunchAgents/com.trueskate.claude-handoff-once.plist"
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

LABEL="${LABEL:-com.trueskate.claude-handoff-once}"
MODEL="${MODEL:-fable}"
EFFORT="${EFFORT:-max}"
CLAUDE_BIN="${CLAUDE_BIN:-$(command -v claude || true)}"
HANDOFF_FILE="${HANDOFF_FILE:-$REPO/tmp/HANDOFF_spin_quality_20260717.md}"

if [ -n "${DELAY_SECONDS:-}" ]; then
  DELAY_SECS="$DELAY_SECONDS"
else
  DELAY_HOURS="${DELAY_HOURS:-5}"
  DELAY_SECS=$((DELAY_HOURS * 3600))
fi

if ! [[ "$DELAY_SECS" =~ ^[0-9]+$ ]] || [ "$DELAY_SECS" -le 0 ]; then
  echo "ERROR: DELAY_SECONDS/DELAY_HOURS must resolve to a positive integer seconds value."
  exit 1
fi

if [ -z "$CLAUDE_BIN" ] || [ ! -x "$CLAUDE_BIN" ]; then
  echo "ERROR: claude binary not found/executable. Set CLAUDE_BIN explicitly."
  exit 1
fi

if [ ! -f "$HANDOFF_FILE" ]; then
  echo "ERROR: handoff file missing: $HANDOFF_FILE"
  exit 1
fi

U="$(id -u)"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
STATE_DIR="$HOME/.claude/scheduled"
PLIST="$LAUNCH_AGENTS_DIR/$LABEL.plist"
PROMPT_FILE="$STATE_DIR/$LABEL.prompt.md"
RUNNER_FILE="$STATE_DIR/$LABEL.runner.sh"
RUN_LOG="$STATE_DIR/$LABEL.run.log"

mkdir -p "$LAUNCH_AGENTS_DIR" "$STATE_DIR"

if launchctl print "gui/$U/$LABEL" >/dev/null 2>&1 || [ -e "$PLIST" ] || [ -e "$RUNNER_FILE" ]; then
  echo "ERROR: existing scheduled job/artifacts found for $LABEL (collision guard)."
  echo "Resolve with: launchctl bootout gui/$U/$LABEL 2>/dev/null || true"
  echo "Then remove:  rm -f \"$PLIST\" \"$RUNNER_FILE\" \"$PROMPT_FILE\""
  exit 1
fi

TARGET_EPOCH=$(( $(date +%s) + DELAY_SECS ))
YEAR="$(date -r "$TARGET_EPOCH" +%Y)"
MONTH="$(date -r "$TARGET_EPOCH" +%m)"
DAY="$(date -r "$TARGET_EPOCH" +%d)"
HOUR="$(date -r "$TARGET_EPOCH" +%H)"
MINUTE="$(date -r "$TARGET_EPOCH" +%M)"
SECOND="$(date -r "$TARGET_EPOCH" +%S)"

cat > "$PROMPT_FILE" <<EOF
Wake-up execution brief for this branch:

1) Start from this handoff context and treat it as the primary task list.
2) Inspect memory files and recent sessions for this repo/workstream.
3) Verify whether the tasks described there were actually completed in this branch.
4) If any task is incomplete, keep building until those tasks are complete.
5) Stay on the current branch; do not merge to main unless explicitly instructed.

Handoff context file path:
$HANDOFF_FILE

--- BEGIN HANDOFF CONTEXT ---
EOF
cat "$HANDOFF_FILE" >> "$PROMPT_FILE"
cat >> "$PROMPT_FILE" <<'EOF'
--- END HANDOFF CONTEXT ---
EOF

cat > "$RUNNER_FILE" <<EOF
#!/bin/bash
set -uo pipefail
export PATH="/usr/local/bin:/opt/homebrew/bin:$HOME/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
# cd to the repo: Claude's permission allowlist is cwd-keyed. Launched from
# launchd's default cwd (/) the session comes up read-only in -p mode — every
# write/exec auto-denied (observed on the 2026-07-17 03:43 wake-up).
cd "$REPO"
"$CLAUDE_BIN" --model "$MODEL" --effort "$EFFORT" --permission-mode acceptEdits \\
  -p "\$(cat "$PROMPT_FILE")" >> "$RUN_LOG" 2>&1
# Cleanup BEFORE bootout, and no set -e: booting out our own label kills this
# script instantly, so it must be the LAST line — the first run bootout'd first
# and left plist+prompt+runner behind. A failed claude run must still clean up.
rm -f "$PLIST" "$PROMPT_FILE" "$RUNNER_FILE"
launchctl bootout "gui/$U/$LABEL" >/dev/null 2>&1 || true
EOF
chmod +x "$RUNNER_FILE"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$RUNNER_FILE</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Year</key><integer>$YEAR</integer>
    <key>Month</key><integer>$MONTH</integer>
    <key>Day</key><integer>$DAY</integer>
    <key>Hour</key><integer>$HOUR</integer>
    <key>Minute</key><integer>$MINUTE</integer>
    <key>Second</key><integer>$SECOND</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>$RUN_LOG</string>
  <key>StandardErrorPath</key>
  <string>$RUN_LOG</string>
</dict>
</plist>
EOF

launchctl bootstrap "gui/$U" "$PLIST"
launchctl enable "gui/$U/$LABEL"

echo "Scheduled: $LABEL"
echo "Target time: $(date -r "$TARGET_EPOCH" '+%Y-%m-%d %H:%M:%S %Z')"
echo "Prompt source: $PROMPT_FILE"
echo "Run log: $RUN_LOG"
