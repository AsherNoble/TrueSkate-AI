# Pico camera-adapter smoke test

This is a first-generation RP2040 Pico USB enumeration/hover diagnostic, not a
gesture executor or collection launcher. Firmware sends no presses, scrolls,
keyboard input or gameplay gestures. Only one test runs per power cycle.

USB exposes one standard three-button relative mouse interface (three report
bytes), boot protocol, interrupt endpoint interval 1 ms, and a 100 mA bus-powered
configuration. It does not expose CDC, storage, or a keyboard. The interval is a
descriptor request; it does not establish iOS/game delivery timing.

After the host mounts the device, the LED slowly blinks for eight seconds. The
cursor then traces two small squares over about 3.2 seconds (80 movement reports
at least 40 ms apart); the LED blinks quickly. After a neutral report, the LED
stays lit and no further reports are emitted. There is a five-second movement
budget. Host loss/suspend or timeout aborts without restarting; a double blink
indicates failure. A solid LED means TinyUSB accepted the reports, not that the
phone rendered them. Confirm visible cursor motion independently.

Build with Arduino-Pico 6.2.0 and its included Adafruit TinyUSB library:

```sh
arduino-cli compile --fqbn rp2040:rp2040:rpipico:usbstack=tinyusb \
  --output-dir /absolute/tmp/pico-smoke-output /absolute/hardware/pico_mouse_smoke
```

Flash the resulting `.uf2` through the Pico's BOOTSEL drive. The mouse-only
firmware requires BOOTSEL for future flashing; there is deliberately no serial
reset interface. Do not flash a Pico W/Pico 2 target without checking its board
and LED pin first.

For the XR2 adapter check, leave collection off, unlock XR2, enable AssistiveTouch,
and connect the Pico's USB to the camera adapter while its Lightning input has
power. Watch for the cursor during the eight-second lead and two-square test.
Capture accessory/power warnings and LED state. Re-powering deliberately runs
one more test. This test works on the Home screen; no True Skate gesture or WDA
recording is required.

References: [Arduino-Pico USB](https://arduino-pico.readthedocs.io/en/latest/usb.html),
[Pico BOOTSEL](https://www.raspberrypi.com/documentation/microcontrollers/pico-series.html),
[Apple pointer support](https://support.apple.com/en-us/111775).
