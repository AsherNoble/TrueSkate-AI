# Pico USB pointer rate and jitter probe

USB counterpart of the Bluetooth hover-rate probe
([HID-POINTER-20261004 section 8](../../research/experiments/HID-POINTER-20261004.md)).
The Pico plays a fixed schedule from its own clock, with no command link:

- home and park before each section;
- 10 passes at 15 ms (first: the Bluetooth comparison);
- 4-count hover passes at 15, 3 and 1 ms (30, 60 and 60 reports each way, slowest first);
- 10 passes at 1 ms;
- a final re-home and re-park;
- a tap, plus one 10-report drag each at 15 ms and at 1 ms, at the clear-floor park point.

It is mouse-only, with the same descriptor as `pico_mouse_smoke`: 1 ms interval, no CDC.

`schedule.py` generates `schedule.h` and `schedule.json`; regenerate both before building.
The firmware:

- **Latches before attaching.** It writes a "ran" latch to the last flash sector before USB attaches, and a build that has already run never attaches again. To run again, rebuild and then reflash: the latch is keyed on the compile time, so reflashing the same UF2 stays locked.
- **Records lateness.** For every event it records the time until TinyUSB accepted the report, in 16 µs units up to about 1 s. Send intervals at 1 ms spacing show how often the host polls.
- **LED:**
  - 1 s blink: waiting or lead (8 s after mount);
  - solid: playing (about 34 s);
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
PYTHONPATH=src python hardware/pico_hover_rate/measure.py movie.mov hardware/pico_hover_rate/schedule.json out/ stats.json
# the Bluetooth baseline, through the same tool (probe JSON as the schedule):
PYTHONPATH=src python hardware/pico_hover_rate/measure.py hover_rate_wifi_on.mov hover_rate_wifi_on.json out-bt/
```

Record it with `scripts/ops/run_xr2_wifi_diagnostic.sh <run> --pico-hover --pico-only --pico-seconds 80`.
Press Enter at the Pico prompt, then plug the Pico into the camera adapter at "PLUG THE PICO IN NOW".

Before running:
- Enable AssistiveTouch with "Perform Touch Gestures", as for the Bluetooth pilot.
- Never connect the Pico to the Mac except in BOOTSEL, because the schedule presses the button.

Press positions assume the Bluetooth gain. `measure.py` measures the USB gain, finds the actual park row, and keeps press crops for visual review.

Measure both links with `measure.py` on one machine (FFmpeg decode; the rig's OpenCV lacks it). Its tracker is refined to sub-pixel accuracy and reads the pointer's polarity from the data, so it differs from `hover_rate_measure.py`. Re-measured this way, the Bluetooth 15 ms passes are 5.3% displaced with Wi-Fi on and 3.0% with it off; the original tool reported about 6% ([evidence](../../research/evidence/HID-POINTER-20261004/update-rate/remeasured-20261008)). The run-11 tunnel needs Wi-Fi, so 5.3% is the like-for-like baseline. A perfectly regular synthetic link scores 0% at every spacing, and parked jitter plus a per-pass in-motion residual are reported as noise. Frozen and catch-up counts only mean something at 15 ms.

At 1 and 3 ms, plain "displaced" exists only when the pass's travel equals its report count, so any rate-dependent gain leaves it empty (None, never 0%). With board stats, "assuming every on-time report arrived" uses each pass's own gain and prints that gain against the 15 ms one. Video cannot separate lost reports from changed gain: below 1x, do not cite it as regularity.
