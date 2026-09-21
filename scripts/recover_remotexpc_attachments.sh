#!/bin/bash
# Inspect or remove XCTest screen-recording attachments through Appium's
# supported RemoteXPC cleanup command. Deletion is always verified by a second
# listing because the underlying private RPC can return success without removing
# an orphaned file.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${TRUESKATE_ENV_FILE:-$REPO_ROOT/.env}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/recover_remotexpc_attachments.sh [--dry-run] [all|xr1|xr2]
  bash scripts/recover_remotexpc_attachments.sh --delete [all|xr1|xr2]

The default is a read-only dry run against both XRs. --delete uses Appium's
official cleanup-videos command, then lists again and exits non-zero if any UUID
survives. A surviving UUID needs the guarded stale-orphan procedure documented
in DEPLOYMENT.md; do not loop --delete or recorder starts.
EOF
}

MODE="dry-run"
TARGET="all"
case "${1:-}" in
  ""|--dry-run)
    TARGET="${2:-all}"
    ;;
  --delete)
    MODE="delete"
    TARGET="${2:-all}"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    echo "Unknown option: $1" >&2
    usage >&2
    exit 2
    ;;
esac

case "$TARGET" in
  all|xr1|xr2) ;;
  *)
    echo "Unknown target: $TARGET (expected all, xr1 or xr2)" >&2
    exit 2
    ;;
esac

if [ ! -f "$ENV_FILE" ]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
set +a

: "${IPHONE_XR_UDID:?IPHONE_XR_UDID is missing from $ENV_FILE}"
: "${IPHONE_XR2_UDID:?IPHONE_XR2_UDID is missing from $ENV_FILE}"
export APPIUM_HOME="${APPIUM_HOME:-/Users/training-server/.appium}"

cleanup_videos() {
  local udid="$1"
  shift
  appium driver run xcuitest cleanup-videos -- --udid "$udid" "$@" 2>&1
}

run_one() {
  local name="$1"
  local udid="$2"
  local before after

  echo "----- $name ($udid): dry-run -----"
  if ! before="$(cleanup_videos "$udid" --dry-run)"; then
    printf '%s\n' "$before" >&2
    echo "$name: could not list attachments; confirm the root RemoteXPC tunnel." >&2
    return 1
  fi
  printf '%s\n' "$before"

  if [ "$MODE" = "dry-run" ]; then
    return 0
  fi

  echo "----- $name: delete -----"
  cleanup_videos "$udid"

  echo "----- $name: verify -----"
  if ! after="$(cleanup_videos "$udid" --dry-run)"; then
    printf '%s\n' "$after" >&2
    echo "$name: deletion could not be verified." >&2
    return 1
  fi
  printf '%s\n' "$after"
  if [[ "$after" != *"Found 0 UUID-shaped attachment"* ]]; then
    echo "$name: one or more attachment UUIDs survived cleanup." >&2
    echo "Stop here. Use the stale-orphan procedure in DEPLOYMENT.md; do not retry in a loop." >&2
    return 1
  fi
}

failures=0
if [ "$TARGET" = "all" ] || [ "$TARGET" = "xr2" ]; then
  if ! run_one "XR2" "$IPHONE_XR2_UDID"; then
    failures=1
  fi
fi
if [ "$TARGET" = "all" ] || [ "$TARGET" = "xr1" ]; then
  if ! run_one "XR1" "$IPHONE_XR_UDID"; then
    failures=1
  fi
fi

if [ "$failures" -ne 0 ]; then
  exit 1
fi
