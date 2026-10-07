# Pico USB pointer rate and jitter probe

USB counterpart of the Bluetooth hover-rate probe
([HID-POINTER-20261004 section 8](../../research/experiments/HID-POINTER-20261004.md)).
The Pico plays a fixed schedule from its own clock, with no command link:

- home and park;
- 4-count hover passes at 3, 1 and 15 ms;
- 10 passes at 15 ms;
- 10 passes at 1 ms;
- re-home and re-park;
- a tap, plus one 10-report drag each at 15 ms and at 1 ms, at the clear-floor park point.

It is mouse-only, with the same descriptor as `pico_mouse_smoke`: 1 ms interval, no CDC.

`schedule.py` generates `schedule.h` and `schedule.json`; regenerate both before building.
The firmware:

- **Latches before attaching.** It writes a "ran" latch to the last flash sector before USB attaches, and a build that has already run never attaches again. Reflash to run again.
- **Records lateness.** For every event it records the time until TinyUSB accepted the report, in 16 µs units up to about 1 s. Send intervals at 1 ms spacing show how often the host polls.
- **LED:**
  - 1 s blink: waiting or lead (8 s after mount);
  - solid: playing (about 30 s);
  - a 100 ms flash every 2 s: done;
  - double flash: failed;
  - 10 Hz flicker: already ran.

```sh
PYTHONPATH=src python hardware/pico_hover_rate/schedule.py hardware/pico_hover_rate
arduino-cli compile --fqbn rp2040:rp2040:rpipico:usbstack=tinyusb --output-dir /abs/out hardware/pico_hover_rate
picotool load -v /abs/out/pico_hover_rate.ino.uf2              # BOOTSEL; no -x, so it does not run on the Mac
# after the run, BOOTSEL again (read-only):
picotool save -r 0x101ff000 0x10200000 stats.bin               # _EEPROM_start from the .map
PYTHONPATH=src python hardware/pico_hover_rate/read_stats.py stats.bin hardware/pico_hover_rate/schedule.json stats.json
PYTHONPATH=src python hardware/pico_hover_rate/measure.py movie.mov hardware/pico_hover_rate/schedule.json out/
```

Record it with `validate_ios_ipv4_recording.py ... --pico-hover --pico-only --pico-seconds 80`.
Plug the Pico into the camera adapter at "PLUG THE PICO IN NOW".

Before running:
- Enable AssistiveTouch with "Perform Touch Gestures", as for the Bluetooth pilot.
- Never connect the Pico to the Mac except in BOOTSEL, because the schedule presses the button.

Press positions assume the Bluetooth gain. `measure.py` measures the USB gain, finds the actual park row, and keeps press crops for visual review.
