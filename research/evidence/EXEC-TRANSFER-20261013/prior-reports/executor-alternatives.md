# 5 · Executor alternatives: XCTest vs Pico USB pointer for curves

Saved by main from the agent's hand-back (subagents cannot write report .md files). Analysis script and output: `lagfit.py`, `lagfit-output.txt` in this folder. M = measured (record + number); C/I = claimed or inferred. Record paths are relative to the worktree's `research/`.

## Recommendation

Keep XCTest for the curved MVP and pursue the design doc's transfer-function route (L0, then L1). The Pico is unlikely (about 20–25%) to beat XCTest on single-contact 300–1200 ms curves. Confidence in this recommendation is about 70%.

## Comparison

| Dimension | XCTest / WDA `ae50404a` | Pico USB pointer through AssistiveTouch (XR2) |
|---|---|---|
| Timing at source | **M** ALIGN-20260913: residual from `submitted_to_ios` to visible onset has SD 13.4 ms, max 28.8 ms, over 28 linear gestures (a ceiling set by 30 fps video). | **M** HID-USB curve-01/02: 1,684/1,684 reports sent, max board lateness 32 µs. This is TinyUSB acceptance, not iOS delivery. |
| Delivery to screen | **M (exploratory)** Round 6: per-clip offsets mostly within ±30 ms, both signs. One clip leads by about 170 ms before its final hold. | **M** run-11 hover reports landing a frame early or late: 13/300 (4.3%) at 15 ms spacing, against 5.3% / 3.0% over Bluetooth; 4/120 at 3 ms; 42% at 1 ms (unusable). |
| Onset latency | **M** Completion callback takes the gesture duration plus 0.22–0.32 s. Host-to-submit SD is 60–140 ms, so labels must use `submitted_to_ios`. Held-out middle marker missed by 4.7 ms and 5.1 ms (M1-CURVED rounds 1 and 3). | **M** No host in the loop. Held-out middle control missed the clock fit by 16.7 ms (batch 1, failed) and 14.2 ms (batch 2, passed). Single human-marked onsets, so not like-for-like with XCTest. |
| Interpolation | **M** CURVE-AUDIT: waypoints under 16 ms apart gave 1/8 Good; 150 ms or more apart gave 20/22 Good. **I** MVP segments of at least 100 ms are in the good range. | **M** 3 ms reports; compiled path matches the intended path within 0.3 pt. Over Bluetooth, a frame carrying one report moved 1.02 ± 0.10 reports (slight smoothing). |
| Holds | **M (post hoc)** Round 6: mid or late holds failed 4/8, against 12/92 for the rest. CURVE-AUDIT: pause paths 11/20 Good. | **M, n=2** pause-600: the cursor stayed still through the 180 ms hold, then the final segment lagged up to 37 pt (fail). pause-150 inconclusive. |
| Positioning | **M** Absolute, arithmetic in `ae50404a`. No acceleration. | **M** Relative only; absolute reports ignored. Gain depends piecewise on each report's count-vector length, independent of spacing within 0.55%. Every contact needs a 1.875 s home-and-park first. |
| Repeatability | **M (weak)** 36/50 paths got the same rating on both phones (device and park confounded). Repeats of one command unmeasured (L1). | **M** Gain and park position reproduce. Repeated curves unmeasured. Over Bluetooth, 20 identical multi-stroke schedules produced at least 4 different tricks. |
| Curve fidelity | **M** Round 6 83/100 Meet; CURVE-AUDIT 66 Good / 21 Minor / 2 Major / 11 Unclear. | **M** Batch 2: 0 pass / 3 fail / 3 inconclusive. Errors up to 88 pt at 150 ms and 20–37 pt at 600 ms. |
| Throughput | **M** 10 dispatches per minute; both XRs usable. | **M** 6 curves plus 3 controls per minute, plus flash and re-plug per batch. **I** About 15 per minute with a host command link not yet built; XR2 only. |
| Operational risk | Low. | High: one adapter, capped Wi-Fi tunnel, manual swaps, a cursor visible in every frame, silent AssistiveTouch gesture failures reported (Apple forum 736100). |

## Offline reanalysis of the Pico batch-2 failure (exploratory, n=6)

- The marked cursor runs ahead of the clock anchored on the orange control onsets by −26 ms (reversal-600) and −32 to −56 ms (150 ms cases).
- One global −26 ms shift raises marks within 0.03 from 65/100 to 75/100.
- That fits the operator's note that the orange trail appears where the cursor was 1–2 frames earlier.
- No constant shift removes the pause-600 post-hold lag.

## Open risks (Pico)

1. A visible cursor in every training frame, absent from human clips: Model 1 may learn it as a shortcut.
2. Executor-specific labels and a new qualification would be needed.
3. The post-hold lag is unexplained.
4. Narrow evidence: XR2, one park, 6 scored clips.

## Minimal Pico curved qualification (proposal; steps 2–3 need rig time and user authorization)

0. Free: measure the cursor-versus-trail phase in the retained batch-1 and batch-2 movies, and re-score with a cursor-anchored clock frozen beforehand.
1. Build a host command link and a cursor mask.
2. Paired XR2 run: 30 MVP commands with 6 repeated 5 times, both executors interleaved, scored with L2 D1–D4 and the 0.03 rule.
3. Gate: the Pico must beat XCTest's Meet rate on the same commands and pass the hold check.

Sources: support.apple.com/en-us/111775; developer.apple.com/forums/thread/652700, /712996, /736100.
