#!/bin/bash
# Clear the accumulated XCTest screen-recording attachments that wedge the recorder
# ("Failed to write file" — see memory xctest-recording-attachments-accumulate). Uses the
# OFFICIAL appium-xcuitest `cleanup-videos` script (remotexpc `XCTestAttachment.delete`,
# the same mechanism the driver uses to auto-delete on stop).
#
# PREREQ: the remotexpc tunnel daemon must be running first:
#   sudo launchctl bootstrap system /Library/LaunchDaemons/com.trueskate.remotexpc-tunnel.plist
# cleanup-videos itself does NOT need sudo (it's a client of the root tunnel daemon).
#
#   bash recover_remotexpc_attachments.sh            # DRY-RUN: verify tunnel + list backlog
#   bash recover_remotexpc_attachments.sh --delete   # clear both phones' backlogs
set -u
export APPIUM_HOME=/Users/training-server/.appium
XR2=00008020-001E759E3A60802E   # iPhone_XR2  (iOS 18.7.2)
XR1=00008020-001D2C843A78002E   # iPhone_XR   (iOS 18.7.6)
MODE="${1:-}"

cv() {  # $1=udid  $2=extra-flags
  appium driver run xcuitest cleanup-videos -- --udid "$1" ${2:-} 2>&1
}

echo "##### remotexpc tunnel check + backlog listing (DRY-RUN) #####"
echo "If you see a 'tunnel'/'RemoteXPC unavailable' error here, the tunnel daemon isn't up yet."
echo ""
echo "----- XR2 ($XR2) -----"; cv "$XR2" "--dry-run" | tail -15
echo ""
echo "----- XR1 ($XR1) -----"; cv "$XR1" "--dry-run" | tail -15

if [ "$MODE" != "--delete" ]; then
  echo ""
  echo ">>> DRY-RUN only — nothing deleted. If the UUID lists look right and there was no"
  echo ">>> tunnel error above, clear the backlogs with:  bash $0 --delete"
  exit 0
fi

echo ""
echo "##### DELETING backlog on BOTH phones #####"
echo "----- XR2 -----"; cv "$XR2" "" | tail -12
echo ""
echo "----- XR1 -----"; cv "$XR1" "" | tail -12
echo ""
echo ">>> Backlogs cleared. Recording should now succeed (auto-delete keeps it clean from"
echo ">>> here, since the tunnel is up). Next: re-probe recording and resume the collectors."
