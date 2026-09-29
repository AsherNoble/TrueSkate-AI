# M1-SCREEN-DRAFT-20260925 — Draft timing screen for the replacement corpus

Status: read-only analysis and a prepared blind check. **No clip was excluded,
moved or relabelled, and no manifest was built.** The screen choice and any
recollection are operator decisions.

## Why a screen is needed

M1-AUDIT and M1-ONSET-VALIDATION showed that a falsely early calibration
detection bends the two-anchor fit. Assuming the true clock rate is ~1.0, the
predicted onset error of a clip is `(rate − 1) × (end_wda − gesture_wda)`
(start anchor implicated) or `(1 − rate) × (gesture_wda − start_wda)` (end
anchor implicated). At extreme rates this matched 22/24 blind labels exactly
and 24/24 within one displayed frame.

## Noise floor

Across parks without the fault, the fitted-rate spread is about ±600 ppm
(5th–95th percentile; e.g. Workshop −535/+579 ppm), consistent with ±1 native
frame of anchor quantisation over a ~55 s span. XR1 Los Angeles is shifted:
median +735 ppm, 95th percentile +13,461 ppm. So Los Angeles start anchors are
probably mildly early even inside the current 0.002 screen.

## Screen options

Script: [calibration_clip_screen.py](../../scripts/inspect/calibration_clip_screen.py),
run read-only on the rig over the 13,592-clip replacement pool plus the
310-clip exact-PTS baseline (13,902 clips). "Clip ≤1 frame" keeps clips in
flagged segments whose predicted onset error is at most one native frame.

| Park | Target | All | Segment-exclude 0.002 | Clip ≤1 frame @0.002 | Clip ≤1 frame @0.0008 |
|---|---:|---:|---:|---:|---:|
| The Workshop | 4,029 | 4,052 | 4,052 | 4,052 | 4,041 |
| SLS 2013 Kansas City | 3,784 | 3,809 | 3,770 | 3,772 | 3,760 |
| Skateboard GB 2024 | 2,017 | 2,020 | 1,973 | 1,975 | 1,965 |
| SLS 2015 Super Crown | 2,009 | 2,018 | 1,977 | 1,979 | 1,964 |
| SLS 2015 Los Angeles | 1,261 | 2,003 | 1,067 | 1,152 | 1,089 |
| **Total** | **13,100** | **13,902** | **12,839** | **12,930** | **12,819** |

Evidence: [0.002](../evidence/M1-SCREEN-DRAFT-20260925/clip-screen.json),
[0.001](../evidence/M1-SCREEN-DRAFT-20260925/clip-screen-0.001.json),
[0.0008](../evidence/M1-SCREEN-DRAFT-20260925/clip-screen-0.0008.json).

Clip-level salvage is small (91 clips at 0.002) because high-rate segments
are usually far outside one frame except at their last gesture. Under any
screen, the historical mixture cannot be met without recollection. With the
strictest option (clip ≤1 frame at 0.0008), the shortfalls are about Los
Angeles −172, Skateboard GB −52, Super Crown −45 and Kansas City −24.

## Open question and prepared check

The error formula is unvalidated between 0.0008 and 0.002, and that is where
the threshold choice matters. A sealed blind 24-clip check is prepared and
**not labelled**.

- **Selector:** [select_midband_onset_validation.py](../../scripts/inspect/select_midband_onset_validation.py),
  draw 4, seed 2509250455. It excludes all audit and earlier validation
  sessions, and every clip is XR1 SLS 2015 Los Angeles.
- **Mid-high** (`1.0012 < rate ≤ 1.002`): all 6 available recordings. Early
  clips are predicted at displayed frame 9 (one at 10); late clips at 8.
- **Mid-low** (`0.998 ≤ rate < 0.9988`): the only available recording.
  Descriptive only.
- **Ordinary** (`abs(rate − 1) < 0.0003`): 5 recordings from the same device
  and park, all predicted at frame 8.

**Numeric criterion (fixed before labelling).** Count early clips whose
labelled first trace is at displayed frame ≥ 9.

- Mid-band lateness is **supported** if ≥4/6 mid-high early clips are late and
  ≤1/5 ordinary early clips are late.
- It is **not supported** if ≤2/6 mid-high early clips are late.
- Anything else is **inconclusive**.

Also report exact-frame agreement with the sealed predictions, and the
late-clip frames. Six early clips give little power; a null result does not
validate the 0.002 screen.

- **History:** three earlier draws were discarded before any viewing or
  labelling (see [hashes](../evidence/M1-SCREEN-DRAFT-20260925/midband-sealed-sha256.txt)).
  Draw 1 predicted too little contrast. Draw 3 used other-park ordinary
  controls, so the park would have revealed the category.
- **Viewer:** `http://100.113.165.56:8772/`, serving only `viewer/`. The export
  downloads as `model1-onset-validation-20260924.json` (reused builder); its seed
  field reads 2509250455.

## Cohort-builder integration (commit `0e73bd6`)

`build_model1_scaling_manifests.py cohort` has an opt-in timing screen:
`--timing-screen-rate R --timing-screen-max-frames F`. Its parameters and
per-park exclusion counts are sealed into the manifest (`timing_screen`). The
default behaviour and fingerprints of unscreened cohorts are unchanged.

Dry run over `model1_linear_replacement_20260922` only (13,592 clips; the
310-clip baseline lives in a separate root). Draft manifests were written only
to the rig tmp directory
`/Users/training-server/trueskate-ai-runtime/tmp/die5-compare-phase2-20260925/draft-manifests/`:

| Screen | Clips | Excluded | Los Angeles | Kansas City | Super Crown | Skateboard GB | Workshop |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 13,592 | 0 | 2,003 | 3,809 | 2,018 | 1,710 | 4,052 |
| 0.002, ≤1 frame | 12,628 | 964 | 1,152 | 3,772 | 1,979 | 1,673 | 4,052 |
| 0.0008, ≤1 frame | 12,517 | 1,075 | 1,089 | 3,760 | 1,964 | 1,663 | 4,041 |

The Los Angeles counts match the standalone screen exactly. None of these
manifests is a training decision.

## Red-team review and corrections (2026-09-25)

An experiment-red-team review (read-only) returned **FIXABLE**. It found the
signs, units, code and shortfall arithmetic correct. Corrections:

1. **"≤1 frame" overstated at 0.002.** Segments inside the threshold are
   scored as zero error. Running the formula at every rate above 0.0006
   ([clip-screen-0.0006](../evidence/M1-SCREEN-DRAFT-20260925/clip-screen-0.0006.json))
   shows how many kept clips are predicted late:

   | Screen | LA kept | LA predicted >1 frame | LA >2 frames | Other parks >1 frame | Other parks >2 frames |
   |---|---:|---:|---:|---:|---:|
   | 0.002 | 1,152 | 74 | 20 | 95 | 0 |
   | 0.0008 | 1,089 | 11 | 0 | 47 | 0 |
   | every rate, exclude >2 frames | 1,250 | — | 0 | — | 0 |

   So 0.002 is too loose for Los Angeles. The 0.0008 screen bounds hidden
   predicted errors at 2 native frames (≈1 displayed slot).
2. **Precision.** The formula was validated only to about one displayed slot
   (67–100 ms, ~2–3 native frames). Clip-level salvage inside flagged
   segments therefore depends on unvalidated precision. The conservative
   alternative is to exclude flagged segments whole: at 0.0008, LA falls to
   957.
3. **Direction ambiguity.** A rate above 1 can also come from a late end
   anchor, which puts the error on late clips; the screen would keep exactly
   those. If both anchors are early by the same amount, the error is
   invisible. Only 8 extreme pairs tested the direction.
4. **Survivor bias.** Calibration rejects rates outside 0.98–1.02, and LA's
   95th percentile is ~13,500 ppm, near that cap. LA's true fault rate is
   understated.
5. **Recollection.** Shortfalls are clean-clip counts. At XR1 LA's ~54%
   pass rate, the LA shortfall of ~172 means ~320 raw LA clips unless the
   settle wait or five-touch detection raises it. The pending
   spin-contamination exclusion is not yet subtracted.
6. **Corpus roots.** The 310-clip baseline is in a separate root. Builder
   totals exclude it, while the standalone table includes it; a final
   manifest must include both. The standalone script silently skips clips
   without v2 calibration, while the builder raises.
7. **Blind check.** The earlier draw used other-park controls, which would
   reveal the category; it was replaced before labelling (see above).

**Current recommendation for the operator:** decide between
- the 0.0008 clip screen (1,089 LA), or
- the 0.0008 whole-segment screen (957 LA),

then recollect LA (with the settle precaution) and small top-ups elsewhere.
The 0.002 screen is not recommended.
