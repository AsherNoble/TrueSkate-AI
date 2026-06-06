#!/usr/bin/env bash
# Install the TrueSkate training supervisor as a per-user LaunchAgent.
#
# Usage:
#   deploy/install_launchd.sh [curriculum_path]
#
# Defaults the curriculum to curricula/360_flip.json. Re-run any time to update
# the plist (it reloads). Uninstall with:
#   launchctl bootout gui/$(id -u)/com.trueskate.training
#   rm ~/Library/LaunchAgents/com.trueskate.training.plist
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CURRICULUM="${1:-$REPO/curricula/360_flip.json}"
LABEL="com.trueskate.training"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"

# Prefer the repo venv python so deps are present.
if [[ -x "$REPO/.venv/bin/python" ]]; then
  PYTHON="$REPO/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi

if [[ ! -f "$CURRICULUM" ]]; then
  echo "Curriculum not found: $CURRICULUM" >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$REPO/logs"

sed -e "s#__PYTHON__#$PYTHON#g" \
    -e "s#__REPO__#$REPO#g" \
    -e "s#__CURRICULUM__#$CURRICULUM#g" \
    "$REPO/deploy/com.trueskate.training.plist.template" > "$DEST"

echo "Wrote $DEST"
echo "  python:     $PYTHON"
echo "  repo:       $REPO"
echo "  curriculum: $CURRICULUM"

# Reload (bootout is fine if not currently loaded).
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$DEST"
launchctl enable "gui/$(id -u)/$LABEL"

echo "Loaded $LABEL. Tail logs with:"
echo "  tail -f $REPO/logs/supervisor.out.log"
