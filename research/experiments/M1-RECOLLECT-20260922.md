# M1-RECOLLECT-20260922 — Replacement linear corpus park plan

Status: planned; collection remains off. XR1's currently loaded park awaits
operator confirmation.

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
| XR1 | **PENDING OPERATOR CHECK** |
| XR2 | Skateboard GB 2024 |

When the operator returns, replace the XR1 placeholder with the observed park
before authorizing collection. A collector park label records provenance; it
does not navigate the game, so each park change must be confirmed on the device.

Collection stops only at segment boundaries and may slightly overshoot a quota.
Preserve surplus clips, then create a deterministic 13,100-sample manifest with
the exact counts above for the controlled training comparison.

