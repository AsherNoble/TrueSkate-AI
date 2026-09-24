# M1-RECOLLECT-20260922 — Replacement linear corpus park plan

Status: second bounded stage and Skateboard GB top-up complete. Both collectors
are stopped; the next park is SLS 2015 Super Crown on both devices.

## Objective

Collect an exact-PTS replacement for the 13,100-sample corpus behind the 80.05%
linear Model 1 result while approximately preserving its aggregate park mix.
The historical device-by-park table below is descriptive, not a collection quota;
the operator clarified that matching park totals across the two XRs is the goal.
The 310 audited XR2 clips from Skateboard GB 2024 count toward that total, so
12,790 additional accepted clips remain.

## Historical mixture

Counts below were read from all 13,100 `meta.json` files in the locally
preserved historical corpus at
`tmp/classical-model-1/corpus/model1_linear_20260902`.

| Park | Samples | Mixture |
|---|---:|---:|
| The Workshop | 4,029 | 30.76% |
| SLS 2013 Kansas City | 3,784 | 28.89% |
| Skateboard GB 2024 | 2,017 | 15.40% |
| SLS 2015 Super Crown | 2,009 | 15.34% |
| SLS 2015 Los Angeles | 1,261 | 9.63% |
| **Total** | **13,100** | **100.00%** |

The historical device allocation was not uniform:

| Device | Park | Samples |
|---|---|---:|
| XR1 | The Workshop | 2,695 |
| XR1 | SLS 2015 Super Crown | 2,009 |
| XR1 | SLS 2015 Los Angeles | 1,261 |
| XR2 | The Workshop | 1,334 |
| XR2 | SLS 2013 Kansas City | 3,784 |
| XR2 | Skateboard GB 2024 | 2,017 |

XR1 contributed 5,965 samples (45.53%) and XR2 contributed 7,135 (54.47%).
Historically, XR1 moved from Workshop to Super Crown to Los Angeles; XR2 moved
from Workshop to Kansas City to Skateboard GB 2024.

## Aggregate park targets

The aggregate historical park counts are the control variable. The current
replacement counts include the preserved 310-clip XR2 Skateboard GB audit
tranche and the bounded top-up completed on 2026-09-24 (Sydney time).

| Park | Historical target | Accepted | Difference |
|---|---:|---:|---:|
| The Workshop | 4,029 | 4,052 | +23 |
| SLS 2013 Kansas City | 3,784 | 3,809 | +25 |
| Skateboard GB 2024 | 2,017 | 2,020 | +3 |
| SLS 2015 Super Crown | 2,009 | 0 | −2,009 |
| SLS 2015 Los Angeles | 1,261 | 2,003 | +742 |
| **Total** | **13,100** | **11,884** | **−1,216** |

Collect approximately 2,009 Super Crown clips across both XRs, then select a
deterministic 13,100-sample manifest that excludes the 793 surplus clips in
other parks. The two devices do not need separate park quotas.

A collector park label records provenance; it does not navigate the game, so
each later park change must still be confirmed on the device.

Collection stops only at segment boundaries and may slightly overshoot a quota.
Preserve surplus clips, then create a deterministic 13,100-sample manifest with
the aggregate historical counts above for the controlled training comparison.

## First bounded tranche launch

The operator authorized collection to 2,000 total strict clips per device. The
audited XR2 baseline is exactly 310 clips, despite the tranche's informal
"300-clip" name. The finite targets are therefore:

- XR1: 2,000 new strict clips;
- XR2: 1,690 new strict clips plus the 310 audited baseline, for 2,000 total.

Release `13554c98b654d10fd30363c99caef6bf9f27efa7` is deployed as a clean rig
worktree; the previous `b60f19d` release remains intact. Before launch, one
simultaneous isolated segment per XR passed strict provenance and unique-command
admission with 11/11 payload clips on each device. All 22 clips decode to 32
frames, have 32 strictly increasing source-relative timestamps, and report
`wda_submitted_two_anchor` timing. Both recorders returned idle.

Production outputs are separate from smoke evidence:

- XR1: `data/model1_linear_replacement_20260922/iPhone_XR/sls_2015_los_angeles`;
- XR2: `data/model1_linear_replacement_20260922/iPhone_XR2/skateboard_gb_2024`.

Each collector enforces its finite strict-admission target between complete
one-minute segments. A persisted monitor sends exactly
`XR1 is at [10–100]% completion` or `XR2 is at [10–100]% completion` at newly
crossed ten-percent milestones. XR2 starts with 10% recorded as already reached,
so its first new notification is 20%; XR1's first is 10%. The first production
segments admitted 11 clips on each device and both collectors began segment 2.

The operator subsequently confirmed XR1's park as SLS 2015 Los Angeles. XR1
was stopped at a completed segment boundary while XR2 continued. All 598 strict
XR1 clips then had the placeholder corrected in clip and segment metadata and
their park directories renamed. Strict count and the sorted command fingerprint
were unchanged (`b8da2ae92e7b34a252ccc0441e96e0b698a2028f9ea4b5c4e6c641ba648e1e13`);
the post-change audit found 598 XR1/SLS 2015 Los Angeles clips, no duplicate
commands and no contamination markers. Collection resumed with the preserved
seed, 2,000-clip target and ten-percent notification state. The rollback bundle
and audit report are retained on the rig under
`tmp/model1-xr1-park-correction-20260922/`.

The first bounded tranche finished at complete segment boundaries with 2,003
strict XR1/SLS 2015 Los Angeles clips and 1,698 new XR2/Skateboard GB 2024
clips. Together with the preserved 310-clip XR2 baseline, the latter park has
2,008 clips. Surplus clips remain available for the final deterministic
13,100-sample manifest.

## Second bounded tranche launch

On 2026-09-23 the operator moved XR1 to The Workshop and XR2 to SLS 2013 Kansas
City, then authorized independent park-directory targets of 4,050 and 3,800
strict clips respectively. Collection continues under release `13554c9` at:

- XR1: `data/model1_linear_replacement_20260922/iPhone_XR/the_workshop`;
- XR2: `data/model1_linear_replacement_20260922/iPhone_XR2/sls_2013_kansas_city`.

Each new directory inherited its device's persisted next seed from the completed
park, preventing the gesture stream from restarting at a park boundary. The
first production audit admitted 10 Workshop clips and 11 Kansas City clips; all
checked clips had 32 increasing source timestamps and accepted calibration. A
combined audit of 3,722 replacement-directory clips found zero exact-command
duplicate groups. XR2's first recording failed during `stop_and_save`, admitted
zero clips, and the bounded collector recovered on the next seed. No milestone
notification process was started for this stage. Runtime paths, initial seeds
and targets are recorded on the rig in
`tmp/model1_replacement_stage2_20260923.json`.

## Stage completion and Skateboard GB top-up

Both second-stage collectors stopped automatically at complete segment
boundaries: XR1 admitted 4,052 Workshop clips and XR2 admitted 3,809 Kansas
City clips. With XR2 then confirmed by the operator in Skateboard GB 2024, one
explicitly bounded segment continued XR2's persisted seed stream and admitted
12 strict clips. The new Skateboard GB directory count is 1,710; with the
preserved 310-clip baseline, that park has 2,020 clips. The new segment passed
the strict corpus audit: correct device/park provenance, no duplicate commands,
32 decodable frames and increasing source timestamps in every clip. Two-anchor
calibration passed. Both collectors stopped, and both WDA recorders returned
idle. The rig audit report is
`tmp/model1-skateboard-gb-topup-20260924-audit.json`.
