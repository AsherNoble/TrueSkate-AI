# M1-LABEL-CONTROL-20260922 — Linear label-window comparison

Status: metadata audit complete; no training or collection performed.

## Question

Does the exact-PTS audit tranche alter clip length, commanded gesture duration,
or the gesture's labelled temporal position relative to the 13,100-sample corpus
behind the 80.05% linear Model 1 result?

## Method

Read every `meta.json` in the locally preserved historical corpus at
`tmp/classical-model-1/corpus/model1_linear_20260902`, whose sealed manifest
reproduces the original 13,100-path checkpoint fingerprint. Read every
`meta.json` in the 310-clip exact-PTS XR2 audit tranche on `training-server`.
No video was decoded and no research partition was exposed to a model.

The onset slot is the first zero-based `frame_times` index whose timestamp is
non-negative. This measures the stored timing contract. It does not claim that
the historical pixels are correctly aligned to that contract.

## Results

| Property | Historical 80.05% corpus | Exact-PTS audit tranche |
|---|---:|---:|
| Samples | 13,100 | 310 |
| Stored frames/timestamps per clip | 32 in 13,100/13,100 | 32 in 310/310 |
| Labelled clip span | 2.2667 s in 13,100/13,100 | median 2.2667 s; 2.2333–2.2834 s |
| First stored time | -0.5000 s in 13,100/13,100 | median -0.4839 s; -0.5000 to -0.4526 s |
| First non-negative timestamp | index 7 in 13,100/13,100 | index 7 in 309/310; index 8 in 1/310 |
| Command duration | 0.300011–1.199968 s; median 0.747842 s | 0.301316–1.199103 s; median 0.746796 s |
| Command-duration mean / population SD | 0.749280 / 0.260024 s | 0.745896 / 0.258500 s |

The historical test partition specifically contains 1,965 commands with
durations 0.300247–1.198274 s, median 0.748051 s, mean 0.746684 s and population
SD 0.259060 s.

The sole new index-8 case is
`iPhone_XR2_20260920_203845/skateboard_gb_2024/sample_000186`. Its index-7
timestamp is -0.0009 s and index 8 is +0.0991 s, so the nominal boundary differs
by 0.9 ms rather than a meaningful change of onset policy. Exact source PTS also
preserve a dropped/irregular source-frame interval instead of inventing a
uniform grid.

The historical metadata always used the synthetic sequence
`-0.5000 ... +1.7667 s`, with the first non-negative timestamp at index 7.
The first 256 historical compact videos previously audited decoded to 31
container frames despite their 32 timestamps. That pixel/timestamp defect is a
known difference in recording quality, not a difference in the intended label
window.

## Decision

No duration or onset randomisation is needed to reproduce these control
variables: the old corpus also used a fixed 32-frame window and a fixed onset
slot, while swipe duration varied uniformly over the same 0.30–1.20 second
range. Keep the exact source timestamps because correcting the old alignment
defect is the purpose of the replacement corpus.

For a strict model comparison, hold architecture, training recipe, corpus size,
command distribution and evaluation samples fixed. The strongest evaluation is
to run both checkpoints on one clean held-out corpus; the historical test split
can additionally be reported for continuity, but it retains the old corpus's
timing and contamination limitations.
