# M1-RECOLLECT-20260922 — Replacement linear corpus park plan

Status: bounded collection active. The operator confirmed XR1 is loaded into
SLS 2015 Los Angeles; XR2 remains in Skateboard GB 2024.

## Objective

Collect an exact-PTS replacement for the 13,100-sample corpus behind the 80.05%
linear Model 1 result while approximately preserving its device and park mix.
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

## Replacement quotas

The quotas preserve the historical device/park allocation. Operational park
order may begin from whichever park is already loaded; the final admitted mix,
rather than chronology, is the control variable.

| Device | Park | Final target | Already accepted | Additional target |
|---|---|---:|---:|---:|
| XR1 | The Workshop | 2,695 | 0 | 2,695 |
| XR1 | SLS 2015 Super Crown | 2,009 | 0 | 2,009 |
| XR1 | SLS 2015 Los Angeles | 1,261 | 0 | 1,261 |
| XR2 | The Workshop | 1,334 | 0 | 1,334 |
| XR2 | SLS 2013 Kansas City | 3,784 | 0 | 3,784 |
| XR2 | Skateboard GB 2024 | 2,017 | 310 | 1,707 |
| **Total** |  | **13,100** | **310** | **12,790** |

Current operator-reported device state:

| Device | Loaded park |
|---|---|
| XR1 | SLS 2015 Los Angeles |
| XR2 | Skateboard GB 2024 |

A collector park label records provenance; it does not navigate the game, so
each later park change must still be confirmed on the device.

Collection stops only at segment boundaries and may slightly overshoot a quota.
Preserve surplus clips, then create a deterministic 13,100-sample manifest with
the exact counts above for the controlled training comparison.

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
