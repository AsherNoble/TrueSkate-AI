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
