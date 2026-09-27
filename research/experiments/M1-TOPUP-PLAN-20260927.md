# M1-TOPUP-PLAN-20260927 — Top up the screened corpus with the settle wait

Status: launched 2026-09-27 (see Launch record below).

## Target

The 80.05% model was trained on 13,100 clips (M1-20260904: 9,170 train /
1,965 validation / 1,965 test). The target is therefore 13,100 good clips (those
passing `corpus-screen-v1`, [M1-CORPUS-AUDIT](M1-CORPUS-AUDIT-20260927.md)),
in the historical park mix ([M1-RECOLLECT](M1-RECOLLECT-20260922.md)).

| Park | Good now | Target | Top-up (good) | Current pass rate | Raw clips needed |
|---|---:|---:|---:|---:|---:|
| SLS 2015 Los Angeles | 938 | 1,261 | 323 | 47% | ~690 (less if settle helps) |
| SLS 2015 Super Crown | 1,898 | 2,009 | 111 | 94% | ~120 |
| The Workshop | 3,927 | 4,029 | 102 | 97% | ~105 |
| Skateboard GB 2024 | 1,936 | 2,017 | 81 | 96% | ~85 |
| SLS 2013 Kansas City | 3,715 | 3,784 | 69 | 98% | ~71 |
| **Total** | **12,414** | **13,100** | **686** | | **~1,070** |

"Good now" includes the 310-clip exact-PTS baseline for Skateboard GB, of which
301 pass the screen.

## Schedule (both XRs in parallel)

Operator-confirmed parks (2026-09-27): XR1 in SLS 2015 Los Angeles, XR2 in
SLS 2013 Kansas City. Production pace was ~9.5 clips per one-minute segment and
~3.4 min per segment including alignment (~4.9 min in Los Angeles). The settle
wait adds ~8 s.

- **XR1:** Los Angeles, ~73 segments ≈ 5–6 h at today's yield.
- **XR2:** Kansas City (~8 segments) → Skateboard GB (~9) → Super Crown (~13)
  → The Workshop (~11), ≈ 2.5 h and three operator park changes. It then moves
  to Los Angeles to share the remaining Los Angeles target, which shortens the
  run to ~4 h.

## Configuration

- **Launcher:** `scripts/ops/mvp_collect_linear.sh` unchanged, plus
  `BASIC_LINEAR_START_SETTLE=1 BASIC_LINEAR_END_SETTLE=1`. The data stay
  two-anchor, so the frozen screen applies unchanged.
- **Output:** a new root, `data/model1_linear_topup_20260927/<device>/<park>`.
- **Seeds:** `BASIC_LINEAR_SEED_FILE` points at each device's latest persisted
  seed file, so gesture streams continue and do not repeat. Run a duplicate
  command audit across all roots.
- **Stopping:** per-park finite targets counted on screened good clips (code
  change 1), so the run stops once the goal is reached rather than at a raw
  count.

## Code changes before launch (branch, tested)

1. **Launcher:** `BASIC_LINEAR_ACCEPTED_SCREEN=corpus-screen-v1` makes
   `accepted_total` count only clips that pass the screen.
2. **Cohort builder:** add `--screen corpus-screen-v1` and multiple roots
   (replacement + baseline + top-up), recording the screen in the manifest.

## Launch checklist

1. **Operator:** approve pushing this branch, so the rig can deploy it as a
   clean release (current release `1b8497e` lacks the settle wait and screen).
2. **Attachment cleanup:** the dry run listed 11 XR1 / 7 XR2 attachments, plus
   possibly one from the failed M1-DIE5-R100 segment. Clean up per
   DEPLOYMENT.md until zero remain.
3. **Deploy:** new release worktree; one bounded segment per XR into `tmp/`
   with the settle flags. Check `start_settle` and `end_settle` in the
   manifest, calibration, strict admission and the screen verdict.
4. **Launch:** XR1 in Los Angeles and XR2 in Kansas City (confirmed). Each
   later XR2 park change is confirmed on the phone by the operator. When XR2
   joins Los Angeles, split the remaining Los Angeles target between the
   devices.
5. **Monitor:** watch the per-park screened counts, the Los Angeles pass rate
   (baseline 47%) and settle skips. The settle wait's effect is observed
   descriptively, not tested: different days are confounded.

## Acceptance

Build the final manifest with the screen. Draw a fresh uniform 100-clip audit
(new seed, same frozen scorer and viewer). The corpus is "13.6k good labels"
only if 100/100 are satisfactory. On any failure, diagnose, fix, and redraw.

## Launch record (2026-09-27)

- **Push and cleanup:** the operator approved pushing the branch and cleaning
  up attachments. XR1 had 12 deleted and XR2 had 7; both re-listed 0.
- **Release:** `080bb97` was created as a detached worktree under
  `trueskate-ai-releases/`, with the usual runtime links. There were no
  dependency changes. Import and syntax checks passed on the rig; the runtime
  venv has no pytest, and the commit passes 369 tests locally.
- **Smoke:** one segment per XR into `tmp/topup-smoke-20260927`, using the
  production flags plus settle, and seeds 2509270501/2.
  - XR1 (Los Angeles): 10/10 strict, both settles settled, calibration
    accepted at +361 ppm.
  - XR2 (Kansas City): 12/12 strict, start settled, end settle timed out at
    4.0 s (the control fired as designed), calibration accepted at −265 ppm.
  - All clips had 32 frames, no fallbacks or errors, and both recorders were
    idle afterwards.
  - **Screen:** XR2 12/12; XR1 0/10. XR1's end latency read −53 ms because
    the device epoch clock stepped +185 ms during the segment (epoch span
    55.228 s against a monotonic span of 55.044 s; video matched monotonic
    within 20 ms).
  - A clock step makes the epoch-based gate reject good segments, never
    accept bad ones. It probably explains most of the ~1% of ordinary corpus
    end anchors that failed. The screen stays frozen for this top-up. A
    monotonic-referenced gate (v2) is a possible later yield improvement that
    would need re-validation.
- **Switch:** the stable link moved from `1b8497e` to `080bb97` with no
  collectors running. `1b8497e` remains for rollback. WDA was not restarted.
- **Collectors:** `scripts/ops/mvp_collect_linear.sh` with
  `BASIC_LINEAR_ALLOW_IDLE_NAVIGATION=1`, `BASIC_LINEAR_BATCH_DIRECT_VIDEO=1`,
  `BASIC_LINEAR_START_SETTLE=1`, `BASIC_LINEAR_END_SETTLE=1` and
  `BASIC_LINEAR_ACCEPTED_SCREEN=corpus-screen-v1`:
  - XR1: Los Angeles, target 323,
    `data/model1_linear_topup_20260927/iPhone_XR/sls_2015_los_angeles`. Seed
    file: the replacement corpus's XR1 Workshop file (next seed 235779486,
    last advanced by Super Crown).
  - XR2: Kansas City, target 69,
    `data/model1_linear_topup_20260927/iPhone_XR2/sls_2013_kansas_city`. Seed
    file: the XR2 Kansas City file (next seed 58036345).
  - Logs: `logs/model1_topup_20260927_{xr1_la,xr2_kc}.log`.

## Collection result (2026-09-27)

| Park (device) | Recordings | Strict | Pass `corpus-screen-v1` | Target |
|---|---:|---:|---:|---:|
| Los Angeles (XR1) | 31 | 331 | 331 | 323 |
| Kansas City (XR2) | 7 | 79 | 79 | 69 |
| Skateboard GB (XR2) | 9 | 90 | 90 | 81 |
| The Workshop (XR1) | — | 109 | 109 | 102 |
| Super Crown (XR2) | — | 113 | 113 | 111 |

- **Every top-up clip passed the screen**, with no calibration rejections.
  Before the settle wait, Los Angeles passed at 47% (different days, so this
  is descriptive).
- **Settle behaviour:** start waits ended at about 2 s, meaning the scene was
  already still. Some end waits timed out.
- **Launch incident:** the first Workshop/Super Crown launch failed before
  touching a device. The remote zsh did not word-split an environment string,
  so the launcher hit an unbound empty array under bash 3.2. It was relaunched
  with explicit variables; each device's seed file skipped one value.
- Both recorders ended idle, with 0 XCTest attachments on either XR.

## Final manifest (2026-09-27)

Code `c0819a2`, staged as a detached worktree under
`tmp/final-manifest-20260927/`. Screened cohorts (`--corpus-screen
corpus-screen-v1`, corpus root `trueskate-ai-runtime`):
- replacement: 12,113;
- exact-PTS baseline: 301;
- top-up: 722.

`park-mix`, with seed 20260927 and the historical quotas, produced
**`model1_linear_good13100_20260927`**: 13,100 clips, fingerprint
`sha256:9fadddee…6013e`.
- **By park:** Workshop 4,029, Kansas City 3,784, Skateboard GB 2,017,
  Super Crown 2,009, Los Angeles 1,261.
- **By source:** 12,082 replacement, 718 top-up, 300 baseline.
- **By device:** XR1 6,371, XR2 6,729.
- No duplicate commands or paths. All 13,100 were re-checked against the
  screen by the audit selector (0 excluded).
