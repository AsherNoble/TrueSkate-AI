# M1-LA-ORIGINALS-20260924 — Retained Los Angeles source recordings

Status: bounded capture complete. The operator loaded both XRs into SLS 2015
Los Angeles. These recordings are diagnostic and are not part of the replacement
training pool.

Four one-minute segment attempts were run per device with the canonical linear
gesture sampler, two fixed centre calibration touches, instrumented WDA timing,
pre-segment reset, foreground/gameplay guards, and foreground strict alignment.
The collector ran from an isolated worktree at release `1b8497e` with only the
opt-in `--retain-mov` change. The active rig release and production corpus were
not modified. Both collectors stopped at `--max-segments 4`.

Seven original `.mov` files were retained: four on XR1 and three on XR2. XR2's
first attempt was lost on `stop_and_save` with `WebDriverException`; the next
three saved and aligned. All seven source files decode with FFprobe, have WDA
timing reports, passed two-control calibration, and emitted 72 strict clips in
total (43 XR1, 29 XR2). Appium sessions expired during some foreground alignment
periods and were reconnected before the next recording; neither collector was
left running.

| Device | Segments with originals | Calibration rates | Start-control scores |
|---|---|---|---|
| XR1 | 0–3 | 1.000037, 0.999977, 1.000341, 1.000539 | 34.17, 32.67, 26.90, 31.46 |
| XR2 | 1–3 | 0.999903, 0.999741, 1.001788 | 31.85, 26.34, 10.82 |

No recording reached the previously investigated Los Angeles high-rate range
(>1.004), so this small capture does not reproduce the severe false-anchor
pattern. XR2 segment 3 has the weakest start-control detection and a modestly
high rate; it is the best retained candidate for direct onset inspection, but
these numbers alone do not establish that its anchor is wrong.

Originals and manifests are on the rig at
`/Users/training-server/trueskate-ai-runtime/tmp/model1-la-originals-20260924/`.
The [capture summary](../evidence/M1-LA-ORIGINALS-20260924/capture-summary.json)
lists every original path, frame count, video duration, calibration fit and
control score. These ignored rig videos have **not** been backed up by Git.

Next analysis: compare the visible first centre touch in each source video
against the detector's selected frame, especially XR2 segment 3, before making
any change to admission or the existing 13,902-clip pool.
