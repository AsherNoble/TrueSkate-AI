#!/bin/bash
# Rebuild the WDA runner at the timing revision with a fresh Personal Team profile.
# Run from the rig's desktop Terminal: SSH has no signing keychain. It builds for a
# generic device only, in WDA's default DerivedData, so the paused services
# supervisor installs and launches it with test-without-building once restored.
# No device touch, install, recording or source change happens here.
set -u
REV=ae50404aac12d9f8c41f6c3fa8776e97975eaef5
WDA=/Users/training-server/WebDriverAgent
PLIST=/Users/training-server/Library/LaunchAgents/com.trueskate.services.plist
OUT=/Users/training-server/trueskate-ai-runtime/tmp/wda-rebuild-$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"

if [ "$(git -C "$WDA" rev-parse HEAD)" != "$REV" ] || [ -n "$(git -C "$WDA" status --porcelain)" ]; then
  echo "REBUILD_FAIL: WDA checkout is not clean $REV"; exit 1
fi
if [ -e /Users/training-server/trueskate-ai-runtime/tmp/RIG-BUSY-curved-recording ]; then
  echo "REBUILD_FAIL: rig busy lock present"; exit 1
fi
echo "$REV" > "$OUT/source-sha.txt"

restore() {
  launchctl bootstrap "gui/$(id -u)" "$PLIST" 2>&1 | tail -1
  echo "services restored $(date)"
}
launchctl bootout "gui/$(id -u)/com.trueskate.services" 2>&1 && echo "services paused $(date)"
trap restore EXIT
for _ in $(seq 1 30); do
  pgrep -f "xcodebuild -project $WDA" >/dev/null || break
  sleep 1
done
if pgrep -f "xcodebuild -project $WDA" >/dev/null; then
  echo "REBUILD_FAIL: a supervisor xcodebuild is still running"; exit 1
fi

echo "building (a few minutes; macOS may ask to allow codesign to use your key)..."
xcodebuild build-for-testing -project "$WDA/WebDriverAgent.xcodeproj" -scheme WebDriverAgentRunner \
  -destination generic/platform=iOS -allowProvisioningUpdates \
  "GCC_PREPROCESSOR_DEFINITIONS=\$(inherited) WDA_TIMING_BUILD_REVISION=$REV" > "$OUT/build.log" 2>&1
status=$?
grep -E "error:|BUILD (SUCCEEDED|FAILED)|TEST BUILD (SUCCEEDED|FAILED)" "$OUT/build.log" | sort -u | tail -8 > "$OUT/summary.txt"
cat "$OUT/summary.txt"
APP=$(ls -td /Users/training-server/Library/Developer/Xcode/DerivedData/WebDriverAgent-*/Build/Products/Debug-iphoneos/WebDriverAgentRunner-Runner.app 2>/dev/null | head -1)
if [ "$status" -ne 0 ] || [ -z "$APP" ]; then
  echo "REBUILD_FAIL: xcodebuild exit $status; log $OUT/build.log"; exit 1
fi
/usr/bin/python3 - "$APP/embedded.mobileprovision" "$OUT/profile.json" <<'EOF'
import json, plistlib, sys
raw = open(sys.argv[1], "rb").read()
start = raw.find(b"<?xml"); end = raw.find(b"</plist>", start)+8
p = plistlib.loads(raw[start:end])
info = dict(name=p.get("Name"), team=p.get("TeamIdentifier"), created=str(p.get("CreationDate")),
            expires=str(p.get("ExpirationDate")), devices=len(p.get("ProvisionedDevices", [])),
            app_identifier=p.get("Entitlements", {}).get("application-identifier"))
json.dump(info, open(sys.argv[2], "w"), indent=2)
print("profile", info["app_identifier"], "expires", info["expires"], "devices", info["devices"])
EOF
echo "REBUILD_OK $OUT"
