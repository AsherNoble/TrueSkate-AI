# HID hardware, spin and recording review

Experiment ID: **HID-REVIEW-20261004**. Date: 2026-10-04.

Continuation of the three interrupted reviews following
[HID-POINTER-20261004](HID-POINTER-20261004.md), based on `dd27a22c`.
The original prompts, saved reviewer context and new reports are preserved at
`/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/claude-review-resume-20261004/`.
The adapter review is research only; the frame-rate review uses existing
recordings. The spin review completed three short XR2 diagnostics under the recovered
bounded authorization. Final guard checks passed. XR1 was untouched and
collection remained off; no production data was admitted.

## Adapter route

The wired route warrants a bounded pilot. A working mouse is considerably more plausible than literal 1 ms touch delivery. These percentages are subjective engineering priors, not measured reliability; ranges express uncertainty. Assumptions: genuine parts, a sufficient charger, correct mouse-only firmware, and an accessible local network.

| Link | Conditional probability | Evidence and limit |
|---|---:|---|
| 1. XR accepts hub, mouse and charging together | Belcompany **70% (50–85)**; VTNIU **50% (30–70)**; genuine Apple adapter + hub **95% (90–98)** | Saved [Belcompany listing](https://www.amazon.com.au/dp/B09QSQJP5J) claims XR, mouse and simultaneous ports; saved [VTNIU listing](https://www.amazon.com.au/dp/B0DDY76NZH) leaves chipset, HID and MFi unclear. [Apple](https://support.apple.com/en-us/111811) explicitly supports hubs, Ethernet and powered peripherals. |
| 2. Pico enumerates and produces AssistiveTouch touches, given working host | **95% (85–99)** | [TinyUSB](https://docs.tinyusb.org/en/latest/examples/device/hid_composite.html) supports Pico HID mice; [Apple](https://support.apple.com/en-us/111775) supports wired mice through Lightning. Exact Pico/XR descriptor remains untested. |
| 3. Clean 1 ms timing | USB-side transport **85% (70–95)**; every report becomes an independent, uncoalesced ~1 ms app touch **10% (2–25)** | Configurable [TinyUSB endpoint descriptors](https://github.com/hathach/tinyusb/blob/master/examples/device/hid_composite/src/usb_descriptors.c) establish capability, not iOS delivery. [UIKit](https://developer.apple.com/documentation/uikit/getting-high-fidelity-input-with-coalesced-touches) normally delivers ~60 Hz/coalesces extra touches. AssistiveTouch's exact processing remains undocumented here. |
| 4. Gain remains usable/recalibratable at 1 ms, given delivery | **85% (65–95)** | Local BLE evidence covers 15–60 ms only. [Apple IOHID source](https://github.com/apple-oss-distributions/IOHIDFamily/blob/main/IOHIDEventSystemPlugIns/IOHIDAcceleration.cpp) includes timestamp/rate-dependent acceleration; it does not establish the XR's active configuration. |
| 5. Simultaneous UE300 Ethernet gives phone/rig IP connectivity | **75% (50–90)** | [TP-Link](https://www.tp-link.com/us/home-networking/usb-converter/ue300/) specifies RTL8153 and desktop support, not iOS. Apple supports selected Ethernet adapters; exact UE300 revision/VID/PID remains unverified. Rig needs another adapter and addressing. |
| 6. WDA, recording and cleanup remain reliable for hours without USB data | **70% (45–85)** | [Apple Xcode](https://help.apple.com/xcode/mac/current/en.lproj/dev3e2f4ee6d.html) supports paired network destinations; [TN3158](https://developer.apple.com/documentation/technotes/tn3158-resolving-xcode-15-device-connection-issues) includes Ethernet. Existing Wi-Fi HTTP success retained USB; it proves neither USB-disconnect survival nor RemoteXPC attachment cleanup. |
| 7. Loaded phone stays charged indefinitely | **85% (65–95)** | [Apple](https://support.apple.com/en-us/111811) documents powering phone/peripherals through charging input. Belcompany/VTNIU usable power and thermal behaviour remain unmeasured. |

For Belcompany: **(a) ~65%** pointer while charging (0.70×0.95); **(b) ~6%** literal non-coalesced 1 ms app touches (a×0.10×0.85); **(c) ~25%** functional wired control + recording for hours (a×0.85×0.75×0.70×0.85), without certifying b. Requiring b too makes c **~3%**. Broad overall ranges: a 45–80%, b 1–15%, c 10–45%. Apple reference raises a to ~90%, not the timing guarantee.

The 85% USB transport estimate and 10% end-to-end estimate are alternative scopes, not independent factors. Other estimates condition on preceding links; hub authentication/power is counted once, sustained charging is incremental. Shared hub/power and network/cleanup risks justify broad ranges, not precise multiplication. Practical finer, more regular frame motion could succeed despite coalescing: my conditional prior is ~60% (35–80), ~35% overall before Ethernet. This is an inference, not evidence that gain precedes coalescing.

Top failures and mitigations:

1. Unsupported accessory, insufficient power or unreliable charging: compare with genuine Apple adapter and a properly powered hub. A truthful ≤100 mA Pico descriptor helps; it cannot guarantee the combined power budget. Title claims do not replace [MFi verification](https://mfi.apple.com/en/faqs).
2. OS batching/gain defeats timing: measure cumulative displacement, frame regularity and press/lift preservation; recalibrate at 1/2/4/8/15 ms. USB polling, pointer rendering and game sampling are separate clocks.
3. WDA survives but recording attachments accumulate: validate paired network transport and RemoteXPC cleanup before any longer run; preserve healthy WDA.

Cheapest de-risk steps: borrow a known mouse/Apple adapter; inspect descriptors and log Pico USB completions (the stock example polls at 5 ms and emits every 10 ms). On delivery, use isolated, bounded hover and five-second recording probes, checking decoded frames, battery trend, retrieval and zero surviving attachments. A 60 fps clip can detect lost motion/bursting, not prove 1 ms touch timestamps. Use a separate touch-logging test app if that stronger claim matters. Amazon rereads were blocked; listing claims come from recovered 2026-10-04 HTML. No hardware, rig, or phone actions were performed.


## Spin and independent touches

AssistiveTouch and spin review, resumed 2026-10-04. Claude's saved reviewer context and scratch were recovered; its previous turn stopped after read-only rig checks, before any live test.

**Result: an XCTest-held spin button and the Bluetooth AssistiveTouch pointer coexist on XR2. A physical finger/pad remains untested.**

My confidence is **95%** that the ordinary AssistiveTouch mouse interface cannot supply two independently positioned, live, programmable contacts. This is a judgment, not a measured probability. The stronger statement “AssistiveTouch is always exactly one finger” is false: Apple explicitly supports 2–5 virtual fingers, with coupled movement, and custom multitouch playback. [Apple's AssistiveTouch guide](https://support.apple.com/en-us/111794)

Pinch/Rotate uses a prescribed two-contact gesture; multifinger drags control their virtual fingers together. A recorded stationary hold plus a flick could plausibly form a fixed custom macro. However, sequential recorded strokes replay simultaneously, and live mouse-plus-custom-hold concurrency and accurate scheduling are unverified. These features do not expose independent contact coordinates/timing to the microcontroller. [Apple's iPhone guide](https://support.apple.com/en-tm/guide/iphone/iph96b21954/ios)

Mouse-button assignments select actions; Dwell triggers an action after remaining still; Drag Lock continues the same movable drag. None documents an additional stationary contact controlled separately from the pointer. [Apple's pointer guide](https://support.apple.com/en-us/111775)

Two mice probably operate one AssistiveTouch cursor (**85% judgment confidence**), but I found no conclusive primary-source iPhone guarantee and did not test a second mouse. Apple offers separate raw `GCMouse` devices to applications that implement their own input handling; that is a different interface from True Skate's existing AssistiveTouch route. [Apple's mouse-gaming session](https://developer.apple.com/videos/play/wwdc2020/10617/)

I ran **three isolated diagnostics**, with collection off, XR1 untouched, no settings/firmware/WDA-service changes and no corpus admission. Pre/post checks confirmed the root RemoteXPC tunnel running, WDA healthy, True Skate foreground, recorder idle, attachments zero and the pointer connected/authenticated/subscribed at 15 ms. Park provenance remains **unverified indoor gameplay scene**, not an inferred park label.

Each run reset the board, recorded at requested 60 fps, and included separate centre timing controls. The identical corrected demo scoop was replayed from the board clock; for the combined run GO followed the 4 s WDA hold call by 1.35 s. The first pointer press followed GO by 2.28 s.

| Condition | Visible result | Exact source frames / duration |
|---|---|---|
| Pointer only | Orange scoop and surviving cursor; `360 POP SHOVE-IT / MANUAL`, then FAILED | 446 / 7.50 s |
| WDA hold only | Spin button glows; overhead gameplay view; grounded board, no trick banner | 553 / 9.28 s |
| Combined | Moving pointer trail while spin button stays glowing; cursor survives; `FAKIE / 360 POP SHOVE-IT / NOSE MANUAL`, then FAILED | 554 / 9.30 s |

The simultaneous contacts appear in combined source frames 340–343 (PTS 5.823–5.873 s). The held-button glow continues after pointer lift. This establishes coexistence in this tested configuration. The changed banner does not establish a specific spin-family trick or replay fidelity. [Evidence still](../evidence/HID-REVIEW-20261004/spin/03-combined/coexistence.png), [original frame](../evidence/HID-REVIEW-20261004/spin/03-combined/coexistence-frame342.png), and recordings: [pointer](</Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/claude-review-resume-20261004/spin/01-pointer/recording.mov>), [hold](</Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/claude-review-resume-20261004/spin/02-hold/recording.mov>), [combined](</Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/claude-review-resume-20261004/spin/03-combined/recording.mov>).

The first run stopped on a frame-count discrepancy: rig OpenCV/AVFoundation enumerated 447 frames; SHA-identical local FFmpeg, rig FFmpeg decoder exhaustion and complete ffprobe all agreed on 446 source frames. The original failure is retained. Subsequent validation used FFmpeg passthrough and required agreement with both ffprobe counts. Median source PTS spacing is 16.667 ms; each recording has four larger gaps. MCU lateness was 0–1 µs, which says nothing about microsecond iOS response. Timing controls were retained; no training alignment/admission or end-to-end latency claim is made.

For the real-pad/human follow-up, repeat the same controls and scoop at confirmed (25,362) pt: pad on at GO+1.28 s, pointer down at GO+2.28 s, pad off at GO+4.50 s; cue a human at those times. Verify separate continuous button glow and pointer trail, surviving cursor and banner. Repeat three per condition, each below 45 s. Record the switch edge with an LED/logic analyser and independently measure first/last recognised screen contact; 60 fps cannot certify microsecond touch latency. The physical pad exercises the digitizer, so this XCTest result is feasibility evidence only. [Appium documents XCTest's private event APIs](https://appium.github.io/appium-xcuitest-driver/latest/guides/input-events/).

Final state: buttons explicitly released (`NOW 0 0 0`, `OK`), gameplay foreground, WDA ready, recorder null, root tunnel running, attachments zero; bridge PID 30867 unchanged. [Final guard evidence](../evidence/HID-REVIEW-20261004/spin/final-state-summary.json). Rig evidence: `/Users/training-server/trueskate-ai-runtime/tmp/hid-pointer/agent-spin-codex-20261004/`.


Curated diagnostic evidence and historical harnesses: [spin evidence](../evidence/HID-REVIEW-20261004/spin/README.md).

## Recording frame rate

**Yes for replay experiments: use XCTest `--fps 60`. Collection can capture 60, but keep the current collection recipe until a separately authorized one-minute validation and temporal-sampling decision.** No phone actions, recordings or collection were performed for this review.

**The game contributes distinct frames.** Reused Claude’s completed 29-video analysis rather than rerunning it. In twenty 60-fps replays, the common moving-game window at 4.0–6.8 seconds contained **3,364 adjacent pairs; none looked duplicated**. Analysis masks orange trails, the AssistiveTouch button and home indicator, excludes HUD tiles, and conservatively subtracts twenty changed tiles per pair to cover both positions of the small cursor. At least **37.4%** of remaining valid tiles changed materially on every pair; median 51.5%. Static hover controls showed at most 8.3% changing tiles before cursor subtraction. This rejects the “30-Hz game plus 60-Hz cursor” explanation for these recordings. It does not certify every park/session or directly instrument the renderer.

**Measured capture and storage.** All originals are 828×1792 H.264. These gameplay recordings are **8.60–8.77 seconds**, correcting the original prompt’s “~30 seconds.”

| Requested rate | Videos | Effective PTS rate | Weighted MB/min |
|---|---:|---:|---:|
| 30 | 6 | 29.77–29.83 | 74.75 |
| 60 | 20 | 59.30–59.65 | 75.19 |

Thus raw bytes rose only **0.6%**, not twofold, in these short comparisons; equal bitrate does not prove equal image fidelity. Per 1,000 recorded minutes, originals would occupy approximately **74.7 versus 75.2 GB**. Forty-eight hours of continuous recording would be 215 versus 217 GB per device; rig wall time is not recorded time. Fixed 32-frame training clips need not grow.

**Gaps and pacing.** The 60-fps replays have 93 long PTS gaps: 92 at ~33.3 ms and one at 50 ms, representing 94 absent 60-Hz slots (~0.9%). Seventy-two occur within the first second, thirteen between 1–3.5 seconds, and eight at 7.66–8.40 seconds. None overlap the audited 4.0–6.8-second motion window. The 30-fps recordings also have 3–4 gaps each at 50 ms. No regular alternate-frame stalls appeared. These observations cannot establish whether recording causes the gaps without a nonrecording/display-timing control.

**Transfer/decode.** Claude’s existing transfer copied 282.5 MB in 87.5 seconds: ~3.23 MB/s, implying ~23 seconds per recorded minute at either rate. Phone-to-rig retrieval was not benchmarked. Local Apple-silicon FFmpeg software decode of one ~8.7-second original per rate took **1.06/1.18 seconds** at 30/60; downscale plus CRF-20 medium H.264 encode to width 512 took **3.40/5.18 seconds**. These bounded two-thread benchmarks do not predict the Intel rig’s runtime.

**Scaling support.** The current wrapper sends only `fps`. [Appium’s XCTest API](https://appium.github.io/appium-xcuitest-driver/latest/reference/execute-methods/#mobile-startxctestscreenrecording) recommends 1–60; it exposes no resolution or bitrate control. [WDA constructs a full-screen request](https://github.com/appium/WebDriverAgent/blob/master/WebDriverAgentLib/Routing/FBScreenRecordingRequest.m). Postcapture scaling saves retained bytes, but cannot reduce phone capture or initial retrieval costs. [MJPEG settings](https://appium.github.io/appium-xcuitest-driver/latest/reference/settings/#mjpegserverframerate) allow a 60-fps ceiling and scaling; [its implementation](https://github.com/appium/WebDriverAgent/blob/master/WebDriverAgentLib/Utilities/FBMjpegServer.m) still takes original-resolution screenshots first. Sustained 60 distinct MJPEG frames is unverified, and the existing consumer stamps host decode time, losing native PTS.

**Pipeline/benefits.** Native onset sampling intervals improve from 33.3 to 16.7 ms; centroid half-frame bias potentially drops from 16.7 to 8.3 ms when raising source capture from 30 to 60. This helps short flicks, phase analysis and stroke extraction; it does not certify better alignment. However, `align_xctest_traces.py` defaults to **32 evenly spaced source-frame indices across 2.3 seconds**; `BasicLinearClipDataset` also defaults to 32. Merely recording 60 does not double Model 1’s temporal resolution. More frames or shorter windows require a deliberate recipe/model study. Keep actual source `frame_times` authoritative; never rebuild timing from frame index or compact MP4 playback. Pass capture rate through alignment/calibration, parameterize remaining 30-fps timing-screen/certification/viewer constants for new recordings, and preserve onset-window/schema/checkpoint contracts. Do not reinterpret historical frame-based evidence at 60. Keep one-minute segments and all admission guards. A hub migration also needs RemoteXPC attachment cleanup/retrieval validation.

Measurements: `measurement-summary.json`, `recording-audit.json`, `pts-gaps.csv`, `decode-benchmark.json`; reproducible offline scripts alongside this report.

Durable measurement summaries and analysis scripts: [FPS evidence](../evidence/HID-REVIEW-20261004/fps/README.md).


## Decisions and next bounded steps

- Retain 60 fps for pointer replay diagnostics, using actual source PTS and exact decoder/probe verification. The current production collection recipe is unchanged. Raising raw frame rate alone does not increase the fixed 32-frame model input.
- The USB hardware remains a pilot proposal. Validate powered mouse operation, then cumulative displacement and frame regularity at 1/2/4/8/15 ms. Validate no-USB recording retrieval and RemoteXPC cleanup before sustained rig use.
- Use the observed WDA/pointer coexistence as feasibility evidence. The next physical spin test needs a human at the phone or the actual pad; use the same controls and independently measure switch/contact latency.

The probabilities above are subjective judgments, and the recordings are small diagnostic comparisons. No hardware purchase, firmware change, paid training, new corpus or production default change was performed. The existing 30-entry journal and append-only archive were preserved.
