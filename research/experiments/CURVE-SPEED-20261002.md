# CURVE-SPEED-20261002 — User-requested long, fast visual probe

The operator requested a fast curve starting diagonally from bottom-left and spanning approximately the screen width, after viewing CURVE-EXEC pilot clips. This separately authorized probe used one XR1 recording on operator-confirmed Inbound: three diagnostic curves and start/middle/end controls. No collector jobs, main/confirmation stages or training were started.

The hand-designed Bernstein coefficients were (0.14,0.82), (0.40,0.62), (0.62,0.23), (0.96,0.38). This extends beyond the seeded experiment's central sampling rectangle to meet the visual request; the convex hull and every quantized segment pass existing global bounds and expanded control exclusions. Horizontal span is 82% of screen width.

Durations were 300, 200 and 120 ms, each with 16 segments. Minimum spacing was 18, 12 and 7 ms; combined compilation bounds were 0.002855, 0.002801 and 0.003035 respectively. Each command used one down/moves/up request. All 51 label positions agree with their command boundaries. Existing cubic/compiler and fake-device scheduling tests passed (34 tests).

The frozen [manifest](../evidence/CURVE-SPEED-20261002/manifest.json) preceded execution. All six WDA requests completed successfully, with no scheduling or recording error. Native video has 1,768 source frames. Post-recording admission **failed**: the original recording contains a gameplay-contamination flag. Preserve this failed attempt; no replacement recording or further device gesture was made. The flag does not establish which gameplay state occurred without human review. Disconnect also logged an expired Appium session after the long offline decode; WDA was not restarted.

The lossless viewer preserves source timestamps using 600-unit movie/track timescales. Jump points use approximate WDA epoch minus recorder startedAt, not an accepted calibration. Independent control inspection also did not establish a passing held-out timing check. Human viewing is primary for this qualitative request; no orange-trail extraction was run and no execution-fidelity pass or actual 120 ms motion claim is made.

Compact outcomes and artifact provenance: [summary](../evidence/CURVE-SPEED-20261002/summary.json). Videos remain isolated in ignored tmp output. Local interactive viewer: http://127.0.0.1:8767/speed/?v=1, with normal playback, speed controls, native frame stepping, looping and downloads. Curve recovery and generation by the models remain the next tranche.
