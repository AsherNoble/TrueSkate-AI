# M1-TIMING-GATE-20260926 — Command-latency gate for calibration anchors

Status: preregistered before the held-out corpus was examined for this
question. Read-only analysis; no clip is excluded and no code path changes.

## Origin (post hoc)

In M1-DIE5-COMPARE and M1-DIE5-R100, all 120 human-labelled start and end
controls appeared 118–174 ms after the WDA submission. The command's video
time is taken as `wda_submitted_epoch_s − started_at_epoch_s`. Every gross
detector error (single-touch and five-touch) fell earlier than 118 ms, some
before the command was sent
([latencies](../evidence/M1-DIE5-R100-20260925/anchor-latency-vs-wda.json)).
The window was derived from those labels, so it needs held-out testing.

## Gate

Accept a calibration detection only if
`onset_video_s − (wda_submitted_epoch_s − started_at_epoch_s)` lies in
**[0.085, 0.210] s**. That is the labelled range widened by about one frame
on each side.

## Held-out test (label-free)

- **Data:** the `model1_linear_replacement_20260922` corpus (1,409 aligned
  segments; different days from the labels). For each segment, the stored
  start/end `onset_video_s` and fitted rate come from any clip's `meta.json`.
  The WDA epochs and `started_at_epoch_s` come from the segment manifest.
- **Proxy:** the fitted-rate anomaly already matched blind human labels in
  22/24 clips (M1-ONSET-VALIDATION). Rates above 1.002 implicate an early start
  anchor, and rates below 0.998 an early end anchor.
- **Segments that failed calibration** emitted no clips and are absent, so the
  worst cases are under-represented.

Criteria, fixed now:

1. **False rejection (mapping accuracy):** in ordinary segments
   (`abs(rate − 1) < 0.0003`), ≥95% of start and ≥95% of end anchors pass.
2. **Sensitivity:** in high-rate segments (`rate > 1.002`), ≥80% of start
   anchors fail the gate early, and ≥90% of end anchors pass.
3. **Low-rate segments** (`rate < 0.998`, n≈12): descriptive only.

The gate is **supported** if 1 and 2 both pass. If 1 fails, the epoch mapping
is not accurate enough in production and the gate is not usable as specified.
If only 2 fails, the gate misses real faults.

Also report, descriptively:
- the latency distribution by device and park;
- the number of clips in gate-failing segments, and their overlap with the
  0.0008 rate screen.

## Script

[timing_gate_corpus_check.py](../../scripts/inspect/timing_gate_corpus_check.py),
run read-only on the rig.

## Results (2026-09-26) — supported

Ran read-only on the rig with the committed script (`4a496c5`), release
`1b8497e`'s Python. Output:
[timing-gate-corpus.json](../evidence/M1-TIMING-GATE-20260926/timing-gate-corpus.json).
All 1,409 segments were readable: 799 ordinary, 90 high-rate, 20 low-rate,
500 other.

| Criterion | Result | Required |
|---|---|---|
| 1. Ordinary starts pass | 797/799 (99.7%) | ≥95% |
| 1. Ordinary ends pass | 789/799 (98.7%) | ≥95% |
| 2. High-rate starts fail early | **90/90** | ≥80% |
| 2. High-rate ends pass | **90/90** | ≥90% |

- **Latency in production:** in ordinary segments, the median is 0.141–0.144 s
  in every device/park group. The 5th–95th percentiles are about 0.124–0.166 s,
  matching the 118–174 ms of the labelled phase 2 controls. High-rate starts
  have a median of −0.172 s (range −0.715 to +0.052 s).
- **Low-rate segments (descriptive):** all 20 have an early end anchor
  (−0.714 to −0.003 s). Five XR1 Los Angeles ones also have an early start.
  Low-rate segments occur on both XRs and in four parks.
- **Beyond the rate screen:** the gate flags 15 segments that the 0.0008
  screen keeps.
  - Several have both anchors early by a similar amount. The rate cannot see
    that fault (red-team point 3 in M1-SCREEN-DRAFT).
  - A few have both anchors late by ~130 ms, e.g. XR2 Kansas City at
    0.271/0.276 s. That pattern suggests a `started_at` mapping error rather
    than a detector error, which is a limitation of the gate.
- **Rate screen beyond the gate:** the 0.0008 screen flags 20 segments the gate
  passes, at rates of roughly 800–2,000 ppm with both anchors inside the window.

Clips kept at segment level (replacement corpus only):

| Park | All | Gate | Rate 0.0008 (segment) | Both |
|---|---:|---:|---:|---:|
| SLS 2015 Los Angeles | 2,003 | 948 | 957 | 938 |
| SLS 2015 Super Crown | 2,018 | 1,948 | 1,927 | 1,898 |
| The Workshop | 4,052 | 3,972 | 4,007 | 3,927 |
| Skateboard GB 2024 | 1,710 | 1,672 | 1,635 | 1,635 |
| SLS 2013 Kansas City | 3,809 | 3,761 | 3,724 | 3,715 |
| **Total** | **13,592** | **12,301** | **12,250** | **12,113** |

Limitations:
- The proxy is the rate anomaly, not new human labels.
- Segments that failed calibration are absent.
- The gate depends on `started_at_epoch_s`, whose own error is small here
  (98.7–99.7% of ordinary anchors pass) but not zero.

## Implications (not yet acted on)

1. **Corpus screen:** the gate is a label-free, anchor-level check that also
   catches equal-early anchors. It could be combined with the rate screen.
2. **Production detector:** searching for the onset only within
   [0.085, 0.210] s after the command would stop the detector from choosing a
   pre-command transient. It would then find the real touch instead of
   rejecting the segment. This needs a code change and a fresh labelled test,
   since the window came from the Phase 2 labels.
