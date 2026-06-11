#!/bin/zsh
# Unattended-rig watchdog for the manual service stack (no launch_services).
#
# Covers the two failure modes observed 2026-06-10/11 that supervised WDA
# loops alone don't fix:
#   1. Stale iproxy tunnels after a USB blip (WDA alive on-device, port dead
#      on the Mac) -> restart that device's tunnels.
#   2. Dead Appium server -> restart it.
#
# WDA itself is NOT restarted here — the per-device `while true; xcodebuild
# test-without-building` loops own that. Run via Monitor so every action line
# surfaces as a notification.
#
# Usage: zsh scripts/stack_watchdog.sh

APPIUM_BIN="$HOME/.nvm/versions/node/v22.22.0/bin/appium"

# name:udid:wda_port:mjpeg_port:appium_port
DEVICES=(
  "XR1:00008020-001D2C843A78002E:8100:9100:4723"
  "XR2:00008020-001E759E3A60802E:8103:9103:4726"
)

typeset -A wda_down
for d in $DEVICES; do wda_down[${d%%:*}]=0; done

while true; do
  usb=$(idevice_id -l 2>/dev/null)
  for d in $DEVICES; do
    parts=(${(s.:.)d})
    name=$parts[1]; udid=$parts[2]; wda=$parts[3]; mjpeg=$parts[4]; appium=$parts[5]

    # Appium health — restart if dead.
    if ! curl -s -m 3 "http://127.0.0.1:$appium/status" >/dev/null 2>&1; then
      echo "[watchdog] $name appium :$appium dead — restarting"
      "$APPIUM_BIN" --port "$appium" >/dev/null 2>&1 &
    fi

    # Tunnel health — only meaningful when the phone is present and a WDA
    # runner exists (otherwise the WDA loop is mid-rebuild; nothing to fix).
    if ! echo "$usb" | grep -q "$udid"; then
      wda_down[$name]=0
      continue
    fi
    if curl -s -m 3 "http://127.0.0.1:$wda/status" >/dev/null 2>&1; then
      wda_down[$name]=0
      continue
    fi
    wda_down[$name]=$(( ${wda_down[$name]} + 1 ))
    if [ "${wda_down[$name]}" -ge 3 ] && pgrep -f "xcodebuild.*$udid" >/dev/null; then
      echo "[watchdog] $name WDA :$wda dead ${wda_down[$name]} checks with phone+runner present — refreshing tunnels"
      pkill -f "iproxy $wda" 2>/dev/null
      pkill -f "iproxy $mjpeg" 2>/dev/null
      sleep 1
      iproxy "$wda" 8100 -u "$udid" >/dev/null 2>&1 &
      iproxy "$mjpeg" 9100 -u "$udid" >/dev/null 2>&1 &
      wda_down[$name]=0
    fi
  done
  sleep 60
done
