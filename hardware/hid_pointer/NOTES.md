# HID pointer pilot notes

2026-10-04, XR2 (iOS 18.7.10), Freenove ESP32-WROVER-E, esp32 core 3.3.12.

- Flashing works only at 115200 baud (`UploadSpeed=115200`); 921600 fails flash verification.
- After the security fix (Just Works bonding with explicit keys), XR2 connects and subscribes to the
  input report (`STATUS connected=1 subscribed=1`). Before the fix it connected but never subscribed.
- Movement and press/drag reports produced no cursor, trail or board movement in screen recordings.
- AssistiveTouch → Devices → Bluetooth Devices lists "TrueSkate Pointer" greyed out with a spinner
  that never finishes (screenshot in this folder). iOS's pointer stack does not accept the device.

Next: compare against a firmware known to work as an iOS AssistiveTouch mouse (e.g. the ESP32-BLE-Mouse
GATT layout: battery service, protocol mode, boot mouse report) and change one difference at a time.

## Handshake log and the security bug (2026-10-04)

`hid_log.py` holds the serial port open (DTR/RTS deasserted so the board is not reset) and logs
every GATT access. iOS's sequence on pairing: connect → MTU 517 → manufacturer → PnP ID ×2 →
manufacturer → HID info → report map → report reference → CCCD write 1. There was **no
authentication step**: the link stayed unencrypted, so iOS left the device spinning.

Cause: on esp32 core 3.3.12 `BLESecurity::setAuthenticationMode(uint8_t)` (Bluedroid) only stores
the auth flags and never sets `m_securityEnabled`, so `BLEDevice::gattServerEventHandler` skips
`startSecurity()` on connect. Fix: use `setAuthenticationMode(bonding, mitm, sc)`.
