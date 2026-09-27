# M1-TOPUP-PLAN-20260927 — Top up the screened corpus with the settle wait

Status: plan only. Nothing deployed or collected.

## Target

Good clips are those passing `corpus-screen-v1`
([M1-CORPUS-AUDIT](M1-CORPUS-AUDIT-20260927.md)). The target is ~13.6k, in
the historical park mix ([M1-RECOLLECT](M1-RECOLLECT-20260922.md)), so that a
deterministic 13,100 manifest can still be drawn for the controlled comparison.

| Park | Good now | 13.6k target | Top-up (good) | Current pass rate | Raw clips needed |
|---|---:|---:|---:|---:|---:|
| SLS 2015 Los Angeles | 938 | 1,309 | 371 | 47% | ~790 (less if settle helps) |
| The Workshop | 3,927 | 4,183 | 256 | 97% | ~265 |
| SLS 2013 Kansas City | 3,715 | 3,929 | 214 | 98% | ~220 |
| SLS 2015 Super Crown | 1,898 | 2,086 | 188 | 94% | ~200 |
| Skateboard GB 2024 | 1,936 | 2,094 | 158 | 96% | ~165 |
| **Total** | **12,414** | **13,600** | **1,187** | | **~1,640** |

"Good now" includes the 310-clip exact-PTS baseline for Skateboard GB, of
which 301 pass the screen. For the historical 13,100 alone, the top-up would
be 686 good clips.

## Schedule (both XRs in parallel)

Production pace was ~9.5 clips per one-minute segment and ~3.4 min per
segment including alignment (~4.9 min in Los Angeles). The settle wait adds
~8 s.

- **XR1:** Los Angeles only, ~84 segments ≈ 5–6 h at today's yield. If the
  settle wait lifts the yield, XR1 finishes early and takes Workshop or Super
  Crown.
- **XR2:** Kansas City → Skateboard GB → Super Crown → The Workshop,
  ~90 segments ≈ 5 h. Three park changes, each confirmed on the phone by the
  operator.

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
4. **Operator:** confirm each phone's park, then launch. XR1 goes first in
   Los Angeles.
5. **Monitor:** watch the per-park screened counts, the Los Angeles pass rate
   (baseline 47%) and settle skips. The settle wait's effect is observed
   descriptively, not tested: different days are confounded.

## Acceptance

Build the final manifest with the screen. Draw a fresh uniform 100-clip audit
(new seed, same frozen scorer and viewer). The corpus is "13.6k good labels"
only if 100/100 are satisfactory. On any failure, diagnose, fix, and redraw.
