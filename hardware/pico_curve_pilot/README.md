# Preloaded USB curve pilot

Bounded XR2 diagnostic for the return to curved Model 1 work. It admits no
training examples. Workload: one gain recording and two six-gesture recordings,
each 60 seconds. No automatic retries or replacements. Collection stays off.

The firmware source is copied **verbatim** from `pico_hover_rate`: mouse-only
descriptor, 8-second mount lead, compile-keyed run-once latch, bounded host-loss
neutralization, and the same 4096-byte timing sector. Only `schedule.h` changes.
Never attach the armed Pico to a Mac except in BOOTSEL. A fresh preparation and
compile are required for a new run; reflashing the same UF2 remains locked.

## Preparation

Use the repository's existing `.venv`; all CLI paths must be absolute, with
outputs under `tmp`. Example variables below are absolute paths; substitute
the worktree and output directory appropriate to the run.

```sh
TASK_REPO='/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/pico-curve-pilot-worktree'
TASK_PY='/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/.venv/bin/python'
TASK_OUT='/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/pico-curve-pilot-build'
TASK_CONFIG='/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/pico-adapter-build/arduino-cli.yaml'
"$TASK_PY" "$TASK_REPO/hardware/pico_curve_pilot/pilot.py" gain --out "$TASK_OUT/gain"
arduino-cli compile --config-file "$TASK_CONFIG" \
  --fqbn rp2040:rp2040:rpipico:usbstack=tinyusb \
  --build-path "$TASK_OUT/gain/compile" --output-dir "$TASK_OUT/gain/output" \
  "$TASK_OUT/gain/pico_curve_pilot"
"$TASK_PY" "$TASK_REPO/hardware/pico_curve_pilot/pilot.py" package \
  --prepared "$TASK_OUT/gain" --uf2 "$TASK_OUT/gain/output/pico_curve_pilot.ino.uf2" \
  --out "$TASK_OUT/stage"
```

Packaging verifies the compiled schedule, RP2040 UF2 family, timing-sector
boundary and compiler-output checksum. Copy the stage into a new rig `tmp`
directory and verify **every** file against `package-manifest.json`. The deployed
rig checkout is not changed. The manifest binds source bytes and firmware; it
does not by itself prove a device executed them.

With the identified Pico in BOOTSEL, preserve the old stats, then load and
verify the packaged UF2 using the rig's picotool. Use `load -v`, **without `-x`
or reboot**, and verify the timing sector is unchanged after loading:

```sh
TASK_PICOTOOL=/Users/training-server/trueskate-ai-runtime/tmp/pico-adapter-20261006/tools/picotool/picotool
TASK_STAGE=/Users/training-server/trueskate-ai-runtime/tmp/pico-curve-pilot-20261008
"$TASK_PICOTOOL" info -a
"$TASK_PICOTOOL" save -r 0x101ff000 0x10200000 "$TASK_STAGE/fw/stats-before.bin"
"$TASK_PICOTOOL" load -v "$TASK_STAGE/fw/pico_curve_pilot.ino.uf2"
"$TASK_PICOTOOL" save -r 0x101ff000 0x10200000 "$TASK_STAGE/fw/stats-after-flash.bin"
cmp "$TASK_STAGE/fw/stats-before.bin" "$TASK_STAGE/fw/stats-after-flash.bin"
```

Before the live handoff, run the staged coordinator with `--prepare` and all
the live arguments below. This checks collection-off, the root tunnel daemon,
pairing and registry conditions without starting recording or WDA. The normal
Wi-Fi, attachment, foreground and decoded-video checks remain required.

## Operator handoff

On the rig's desktop Terminal, with XR2 in The Workshop and unchanged
AssistiveTouch settings (including Perform Touch Gestures):

```sh
/Users/training-server/trueskate-ai-runtime/tmp/pico-curve-pilot-20261008/run_xr2_wifi_diagnostic.sh \
  gain-01 --pico-hover --pico-only --pico-seconds 60 \
  --pico-program /Users/training-server/trueskate-ai-runtime/tmp/pico-curve-pilot-20261008/pico-program.json
```

Approve the administrator prompt. Unplug the Pico from the rig and hold it in
hand. At **PICO STEP NEXT**, confirm the stated setup and press Enter in that
Terminal. Plug it into XR2's powered USB adapter only at **PLUG THE PICO IN NOW**.
That prompt is immediate in Terminal and speech; ntfy sends in the background
and may arrive later. Its run name distinguishes delayed notifications. Capture
starts before the prompt, and the program begins 8 seconds after USB mount.
After capture and analysis the coordinator says **Run complete** or **Run aborted**,
waiting a bounded time for speech before the wrapper restores volume/mute.
Completion is not a curve-fidelity verdict. Check `report.json`, including
analysis/cleanup errors, before reviewing. The wrapper propagates failures.

Return the Pico to the rig in BOOTSEL and read the last-sector timing receipt
without executing or flashing. Preserve failed/partial stats and movies as well:

```sh
"$TASK_PICOTOOL" save -r 0x101ff000 0x10200000 "$TASK_STAGE/fw/stats.bin"
# On the development Mac, after copying the stats and capture directory:
"$TASK_PY" "$TASK_REPO/hardware/pico_curve_pilot/pilot.py" read-stats \
  --program "$TASK_OUT/gain/program.json" --stats "$TASK_OUT/gain/stats.bin" \
  --out "$TASK_OUT/gain/receipts.json"
```

## Gain and curve generation

The button-up gain program takes 25.322 seconds and has 971 events. It tests
positive/negative lengths 1, 2, 3, 4, 5, 8, 16 plus vertical and diagonal
directions, at 15 and 3 ms. Each pass has stationary before/after windows.
Use `measure-gain --program --receipts --movie --capture-report --out --play-origin-s`
to measure them automatically. It reuses the run-11 circle finder, requiring ≥90%
tracked frames per window, ≤1 pt stationary spread and reproducible home/park.
It retains all detections, source PTS and 74 compact crops. The resulting
`profile.json` can be passed to `curves --profile`. Failed tracking preserves
`evidence.json` and `failure.json` without generating a profile or launching a
replacement recording.

Use `review` with those same options to inspect the native-frame windows or
prepare a manually confirmed measurement. The approximate play origin is only for
locating stationary windows: find the first horizontal gain pass and subtract
1.875 seconds. It supplies no curve-label clock. All output options above take
absolute paths. The raw movie stays outside the web root.

For manual measurement, serve the generated viewer temporarily, open it in Google Chrome, and mark at
least three cursor centres per stationary window. Confirm stability; changing
frames or origins to conceal moving endpoints is not allowed. Each mark records
source frame, position and measurement uncertainty. Export the review and shut
down its server once the exported path is returned. Import with:

```sh
"$TASK_PY" "$TASK_REPO/hardware/pico_curve_pilot/pilot.py" fit-gain \
  --bundle "$TASK_OUT/gain-view-bundle.json" --export "$TASK_OUT/gain-review.json" \
  --media-root "$TASK_OUT/gain-view" --out "$TASK_OUT/usb-gain.json"
"$TASK_PY" "$TASK_REPO/hardware/pico_curve_pilot/pilot.py" curves \
  --profile "$TASK_OUT/usb-gain.json" --batch 1 --out "$TASK_OUT/curves-1"
```

Gain fitting requires all 36 passes, validated endpoints with ≤2 pt uncertainty,
direction agreement within 0.5 pt/report, monotonic radial profiles at both
rates, and reproducible home/park. It rejects inadequate evidence. It does not
use the observed curves as labels or fit gain from them. Each curve program
uses 3 ms reports, feedback from accumulated predicted position, and count
vectors of length ≤16 outside homing. The Bluetooth planner stays at 15 ms.

Compile/package batches 1 and 2 separately using the preparation commands,
changing the prepared/output paths, then stage and load each in BOOTSEL. A real
measured USB profile is required before either program can be prepared for use.
Synthetic profiles belong only to offline tests. Each batch contains 1684 events
over 33.816 seconds and fits the existing sector; its capture is still 60 seconds.

Cases are frozen at normalized centre (0.50, 0.48), horizontal scale 0.10,
vertical scale 0.06: arc, S, reversal, and pause/change-speed at 150 and 600 ms,
with all four 150 ms cases repeated. Batch 1 runs the four fast cases plus slow
arc/S; batch 2 repeats the fast cases plus slow reversal/pause. Each contact is
re-homed/positioned, presses still, moves continuously, then lifts in a separate
still report. Start/middle/end Pico controls are separate contacts. Entire
intended and predicted paths avoid the expanded controls. No gameplay reset or
settled-board requirement is imposed on the diagnostic curves.

## Timing review and scoring

For each curve batch, first build `review` with `--play-origin-s` to locate the
three control clips. Mark first visible contact after a clear previous frame,
export, and import using `fit-controls --bundle --export --media-root --out`.
Timing mode only needs one touch-down mark per clip and confirmation of its
clear preceding frame. Marking pauses playback and shows a green **Touch-down
marked here** button; **Clear touch-down mark** removes that boundary and its
confirmation. Position uncertainty is hidden in this mode. It is used later
for contact-centre measurements, as the radius in logical points within which
the true visible centre could lie around the clicked point.
The first/end onset intervals define **one affine map** from Pico accepted-report
times to native movie PTS; the middle is held out. Onset interval midpoints and
their uncertainty are retained. A middle discrepancy beyond one native frame,
missing boundary frames or incomplete board receipt prevents qualification.
Pico receipt times mean TinyUSB acceptance, not independently observed iOS delivery.

Rebuild `review` with the same program/receipt/movie/report and
`--controls-bundle --controls-export --controls-media` to produce curve clips.
The target is one moving unfilled 10-logical-point-radius ring evaluated at
source PTS. O hides it; arrows step native frames. A translucent logical-point
grid can be toggled independently: each small square is 1 × 1 logical point,
with stronger lines every 5 points. It shares the image/mark transform and does
not intercept clicks. Position uncertainty is a radius around the clicked centre;
use the grid to estimate that radius. Use the browser's existing zoom if needed. The ring follows the intended
trajectory, not observed positions or an individually optimized phase. Mark
contact positions and confirm them against the orange trail; an overlay circle
alone is not proof of in-game contact. Unknown positions/boundaries stay unknown.
Export, stop the viewer, then use `score --bundle --export --media-root --out`.

The scorer requires ≥90% evaluable planned contact frames, every reliable frame's
normalized error **plus measurement/clock uncertainty ≤0.03**, duration error plus
boundary uncertainty ≤0.10 s, confirmed contact boundaries, and no unexplained
contact interruption. It also reports logical-point error, onset/lift error and
board lateness. Definite deviations fail; ambiguous measurements are inconclusive.
At 150 ms the reversal can exceed the position tolerance within half a 60 Hz
frame even under ideal execution; it remains the operator-approved stress case
and may be unqualifiable at this measurement resolution.

`aggregate --first --second --out` requires the exact twelve distinct cases with
the same measured USB profile. Only twelve passes advance to a broader held-out
audit. No pilot outcome authorizes bulk collection or training. A failure does
not establish jitter as the cause: inspect compilation, directional gain, board
timing, alignment and visibility separately. All media, stats and prepared firmware
stay under `tmp`; compact eventual results attach to HID-USB-20261006. The usual
five-touch Model 1 calibration requirements are unchanged: the operator approved
Pico-only controls for this diagnostic only.
