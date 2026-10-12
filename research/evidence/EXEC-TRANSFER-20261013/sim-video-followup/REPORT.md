# Simulator schedule versus physical round-6 trail — 2026-10-13

**Transfer is inconclusive.** Independently measured simulator schedules improve
the primary selected moving-region residual from **18.79 to 13.00 native px**,
but only three held-out recordings contribute, one gets worse, and the result
reverses under the prespecified display-sampling bracket. The real videos do not
yet support adopting the simulator timing law as a physical-XR law. They also
do not reject its easing mechanism. Collection qualification is unchanged.

This was laptop-only, offline and free. No simulator was operated by this
analysis; the sibling XCTest investigation supplied its completed measured
schedules. No rig, device, SSH, cloud, production code or prior outputs changed.

## Frozen comparison and coverage

[PLAN-FROZEN.md](PLAN-FROZEN.md) was written before extracting new head positions
or fitting alignment. All 100 round-6 commands match the simulator export:
maximum numerical difference **5.7e-14 logical pt**. Each schedule includes the
actual measured segment timing and touch positions from five repeats; this is
not merely a fitted Bézier curve. Coordinates are converted from logical pt to
native pixels by multiplying by two (828×1792 video).

The split alternates whole recordings within the five source-prefix pairs:
development first IDs 51/71/91/111/131, held out 61/81/101/121/151. Frames from
one recording never cross the split. This does not establish unseen-device or
independent-session generalisation.

The frozen visibility screen scans the raw cross-section profiles of all 100
clips. It finds a unique bright-to-dark front on a clear, straight, non-retracing
moving stretch, subtracts a pre-onset background, and censors vertices,
endpoints, stationary locations, texture-proxy hits and noisy/inadequate
corridors. It does not use either predicted location to choose the front.
The centre estimate is the measured front minus an assumed cap radius, along
the current path. Perpendicular localisation is not measured independently.

At primary excess threshold25 and cap29px, **27/4,872 screened active frames
(0.55%) from six clips** qualify: eight development frames in three clips from
two recordings, and 19 held-out frames in three clips from three recordings.
All selected frames are before commanded release. Threshold30 adds two frames
from clip90. These screening counts are not estimates of video failure rates.

Before fitting, all earliest/middle/latest frames of every selected run were
visually inspected with source context and enlarged crops. Every selected run
showed a clear moving warm head away from the board and salient scenery edges;
none was manually excluded. [visual-review.json](visual-review.json) records
each judgment; [visual-01.jpg](visual-01.jpg) and
[visual-02.jpg](visual-02.jpg) show the evidence. Most clips remain unidentified
under this deliberately strict measurement method; the screen was not relaxed
after seeing results.

## Forward prediction result

Each model fits **one global time shift on development recordings only**.
The objective first takes each clip's median Euclidean positional residual,
then averages clips within recording and averages recordings equally. No
per-clip shift, timing inversion, stretch, or physical trajectory fit is used.
Positive signed residual means the prediction is ahead along the path.

The primary linear and simulator fitted shifts are −3ms and −2ms respectively
(`prediction_time = movie_command_time − shift`). These small values are
nuisance alignments, not measured touch-to-display latencies.

| Clip | Split | Frames | Linear residual px | Simulator residual px |
|---|---|---:|---:|---:|
| 93 | Development | 1 | 1.53 | 0.90 |
| 99 | Development | 2 | 61.65 | 12.25 |
| 118 | Development | 5 | 4.47 | 26.81 |
| 61 | Held out | 3 | 15.74 | 1.34 |
| 121 | Held out | 1 | 31.43 | 33.29 |
| 153 | Held out | 15 | 9.21 | 4.37 |

The development recording-balanced residual is 18.03→16.69px; held out it is
18.79→13.00px, an apparent **5.80px improvement**. Two of three held-out
recordings improve. With no fitted shifts, held-out residual is 13.85→11.07px
(2.78px improvement). A descriptive recording bootstrap interval for the primary
improvement is −1.86..14.40px; **three clusters are too few for a reliable
population confidence claim**. Frames are not independent replicates.

Clip153 is the most informative qualitative stretch: 15 consecutive frames,
98–331ms on one long leg. Its linear signed residual changes from positive to
negative, while the simulator schedule flattens that pattern. This is compatible
with easing. It does not independently identify delivered touches or app
rendering. Clip61 supplies only three early frames, and clip121 only one frame;
neither identifies a complete waveform. The conflicting development clips
99 and118 make a universal alignment especially weak. Clip99's selected frames
are moving after its initial hold; they say nothing precise about hold duration.

See [forward-residuals.png](forward-residuals.png) and
[heldout-recordings.png](heldout-recordings.png).

## Why the conclusion remains inconclusive

The full prespecified threshold20/25/30 × cap27/29/31px sensitivity retains
**3.24..6.94px improvement when simulator samples are linearly interpolated**.
But treating delivered touch samples as held until the next sample, with its
global shift still fitted only on development, yields **11.83..17.52px worse
held-out residual**. At primary25/29 it is 18.79px linear versus34.19px simulator;
the simulator's learned shift becomes−16ms. The sparse development observations
cannot resolve rendering phase, and a phase that fits them fails to generalise.
The held-sample result is a sensitivity calculation, not proof of how True
Skate consumes touches. [sensitivity.png](sensitivity.png) shows the reversal.

Moving either primary fitted alignment by one native movie frame (16.7ms)
changes held-out errors to 19.64..51.02px for linear and25.59..52.07px for
simulator, larger than the apparent advantage. This is an illustrative timing
sensitivity, not a certified calibration-error bound. Existing movie onset is
a marker-calibrated command clock, whereas simulator zero is observed touch-B.
Physical scheduling, marker phase, app update and rendering are not separately
observed.

The cap27..31px bracket came from stationary held-cap excess25 edges. Its
transfer to moving heads is unproven, and thresholds20/30 need not share that
radius. Broad illumination gradients, asymmetric motion bloom and age-dependent
trail width/fade remain measurement risks even on visually clean heads. The
method therefore compares model predictions to a cap-corrected visible edge,
not to independently logged physical touch centres.

Holds, endpoints, corners and spatially repeated paths are not rescued by these
numbers. **Clip67 contributes no measurements.** Its alleged170ms lead remains
invalid evidence because of retracing/old glow, endpoint scoring and deck
interference; this analysis makes no hold-collapse claim. No future darkness
beyond a finite path is scored.

## Reproducibility and next implication

[study.py](study.py) has separate `extract` and `fit` stages; the visual review
was saved between them. `results.json` contains all18 prespecified sensitivity
settings, `primary-rows.json` the forward XY predictions/observations, and
`selection.json` the counts. [provenance.json](provenance.json) hashes the plan,
manifest,100 raw-profile files and simulator export. [verify.py](verify.py)
independently checks exact commands, split disjointness, seven actual video
hashes and primary aggregation; all checks passed in
[verification.json](verification.json). The generated charts were visually
inspected. Source media and prior reports remain read-only.

Run with the repository `.venv/bin/python` and absolute script path: `study.py
extract`, inspect/write `visual-review.json`, then `study.py fit` and `verify.py`.
Re-running extraction should retain the frozen plan and complete visual review
before fitting. Existing outputs preserve this run.

The useful follow-on is to keep easing plus measured segment scheduling as an
explicit simulator hypothesis, and make any offline compensation study report
display-phase sensitivity. These videos alone cannot decide a physical timing
correction. More fitting of these same sparse edges would not solve the missing
touch/render clock or certify physical holds/corners.
