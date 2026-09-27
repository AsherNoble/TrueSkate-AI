# M1-CORPUS-AUDIT-20260927 — Blind random audit of the screened corpus

Status: screen frozen and audit preregistered before selection. No clip is
moved or deleted; the screen only decides what enters future manifests.

## Operator acceptance rule

A corpus counts as "good labels" when a uniform random sample of 100 clips
is entirely satisfactory. If every clip passes, the bad-clip rate is below
~3% (rule of three, 95%).

## Frozen screen: `corpus-screen-v1`

Defined in code (`trueskate_ai.data.timing_screen.passes_corpus_screen_v1`).
It applies to whole two-anchor segments. A segment is **excluded** if either:

1. `abs(rate − 1) > 0.0008` (M1-SCREEN-DRAFT), or
2. any calibration detection lies outside **0.085–0.210 s** after its
   control's WDA submission (`wda_submitted_epoch_s − started_at_epoch_s`;
   M1-TIMING-GATE).

On `model1_linear_replacement_20260922` this keeps 12,113 of 13,592 clips
(Los Angeles 938 of 2,003).

## Audit 1: the screened replacement corpus, before any top-up

- **Sample:** 100 clips drawn uniformly from all clips that pass
  `corpus-screen-v1`. Seed 2509270100 with
  [select_corpus_audit.py](../../scripts/inspect/select_corpus_audit.py). The
  310-clip exact-PTS baseline is a separate root and is not in this audit.
- **Blinding:** the viewer shows no park, rate, expected frame or overlay.
  Expected frames go to a separate `expected.json`, sealed by SHA-256 before
  viewing.
- **Task (unchanged viewer):** for each clip, mark the first displayed frame
  where a new swipe trace is visible. Press U if it is unclear, there is no
  trace, or anything else makes the clip unacceptable, and add a note.
- **Satisfactory:** labelled (not U), and the labelled frame is within ±1
  displayed frame of the expected frame. The expected frame is the first stored
  frame with a non-negative time; in earlier checks this was displayed frame 8.
  One displayed frame is ~67–100 ms, about 2–3 native frames.
- **Pass:** all 100 satisfactory. Exact matches are reported as a secondary
  measure.
- **Scorer:** [score_corpus_audit.py](../../scripts/inspect/score_corpus_audit.py),
  frozen now.

## Decision rule

- **Pass:** the screen is adequate for the existing clips; top up to ~13.6k
  good clips per park mix and run a fresh 100-clip audit on the final corpus.
- **Fail:** diagnose the failed clips' cause, fix the screen or collection,
  then draw a **new** 100 (never re-check the same sample).

## Audit 1 ready (2026-09-27)

- **Draw:** rig code `e456e27`. The frozen screen kept 12,113/13,592 clips.
  The 100-clip draw contains Kansas City 37, The Workshop 33, Los Angeles 12,
  Super Crown 11 and Skateboard GB 7.
- **Sealed hashes:**
  [audit1-sealed-sha256.txt](../evidence/M1-CORPUS-AUDIT-20260927/audit1-sealed-sha256.txt).
- **Viewer:** `http://100.113.165.56:8773/`, serving only `viewer/`;
  `expected.json` returns 404. The export downloads as
  `model1-corpus-audit-2509270100.json`.

## Audit 1 result (2026-09-27) — pass

The operator labelled all 100 clips (0 unclear, 0 notes). Both sealed hashes
were verified before scoring. Evidence:
[score](../evidence/M1-CORPUS-AUDIT-20260927/audit1-score.json),
[labels](../evidence/M1-CORPUS-AUDIT-20260927/audit1-labels.json),
[expected](../evidence/M1-CORPUS-AUDIT-20260927/audit1-expected.json),
[selection](../evidence/M1-CORPUS-AUDIT-20260927/audit1-selection.json).

- **Satisfactory:** 100/100, so the bad-clip rate among screened clips is
  below 3% (95%).
- **Exact frame:** 97/100. The three off-by-one clips:
  - Los Angeles XR1 at +553 ppm, one frame early;
  - Super Crown XR1 at −401 ppm, one frame early;
  - Kansas City XR2 at +19 ppm, one frame late.
- **Next, per the decision rule:** top up to ~13.6k good clips in the park
  mix, collect under the same screen, then run a fresh 100-clip audit on the
  final corpus.

## Audit 2: the final 13,100 manifest (2026-09-27)

This is the same frozen task, satisfactory rule, pass rule and scorer as
audit 1, on a fresh draw.
- **Draw:** seed 2509270200, from `model1_linear_good13100_20260927`
  ([M1-TOPUP-PLAN](M1-TOPUP-PLAN-20260927.md)). By park: Kansas City 30,
  Workshop 26, Skateboard GB 20, Super Crown 14, Los Angeles 10.
- **Sealed hashes:**
  [audit2-sealed-sha256.txt](../evidence/M1-CORPUS-AUDIT-20260927/audit2-sealed-sha256.txt).
- **Viewer:** `http://100.113.165.56:8774/`; the export downloads as
  `model1-corpus-audit-2509270200.json`.

The corpus is "13,100 good labels" only if all 100 clips are satisfactory.
