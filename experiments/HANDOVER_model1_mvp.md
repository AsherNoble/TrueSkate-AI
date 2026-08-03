# Handover — Model 1 MVP (stationary-touch prototype)

Session date: 2026-07-21. Branch: `feature/behavourial-cloning`.
Plan file: `/Users/ashernoble/.claude/plans/quiet-sauteeing-hammock.md`.

## ⚠️ RIG IS DOWN — read first

The rig (`training-server`, reach via `ssh training-server@training-server`) has **no
working WebDriverAgent on either phone**. Both WDA ports (8100 / 8103) are dead;
`com.trueskate.services` fails `xcodebuild` **exit 65** — the free-team provisioning
profile expired (team `BK75JV5YK4`, Apple ID `asherthenoble@outlook.com`).

**I caused this**: I ran `launchctl kickstart -k gui/$(id -u)/com.trueskate.services` to
test a latency hypothesis. WDA had 5d22h uptime and was working — a *running* WDA
survives signing expiry indefinitely (signing is only needed to *rebuild* it), so the
restart converted a working rig into a down one. See memory
`wda-signing-free-team-7day-expiry` (updated this session with this exact lesson).
**Do not restart `com.trueskate.services` without first confirming signing is valid.**

**Recovery (hands-on at the rig, needs 2FA — not SSH-scriptable):**
1. Xcode → Settings → Accounts → remove and re-add `asherthenoble@outlook.com`.
2. Keep `DEVELOPMENT_TEAM` = `BK75JV5YK4` (do NOT repoint to the cert OU string).
3. `launchctl kickstart -k gui/$(id -u)/com.trueskate.services`.
4. If relaunch misbehaves, kill stale xcodebuild procs still bound to XR2's UDID
   (`00008020-001E759E3A60802E`) — they predate the restart and can block a device.

SLS collection on BOTH phones was intentionally paused for this MVP work (Asher's call).

## Goal

Fastest possible Model 1 (trace-extractor) MVP to isolate ONE question: is Model 1's
ceiling (34.78% per-frame, ~5% stroke recovery) an architecture/pipeline limit, or an
upstream data/label problem? A **stationary touch** (hold/tap) in **The Workshop** park
has an unambiguous (x,y) target, a known onset+liftoff, and a clean static background —
no direction/speed ambiguity. If Model 1 can't near-ceiling on THAT, the problem is
upstream and more collection is wasted. If it nails it, architecture is exonerated.

Decisions taken: holds + a tap arm (80/20); new Modal volume `trueskate-mvp`,
video-encoded; both phones on the MVP (SLS paused). Asher's steer, mid-session: a fully
static trace is direction/speed/onset-ambiguous — do NOT assume "frames-with-a-trace"
are ideal Model 1 inputs; holds are valued for the onset+liftoff *lineage*, not just position.

## What was built (committed on `feature/behavourial-cloning`)

| commit | what |
|---|---|
| `7af2cc0` | hold/tap `GestureSample` kinds + `sample_hold`/`sample_tap` + `static_frac` in `sample_mixture` (`src/trueskate_ai/data/gesture_sampling.py`); collector `_execute` branch; `_static_schedule` label branch → `_TouchInterval(constant_xy=…)` (`temporal_trace_dataset.py`) |
| `cf40f99` | aligner start-anchoring: `gv = t_call_start + Δ`; per-kind measured Δ (`_DELTA_BY_KIND`); emits `gesture_start_monotonic`; `--anchor start\|end` |
| `c4c829c` | optional per-sample `frames.mp4` (h264) instead of N PNGs; dataset loader reads either transparently (`_video_path`/`_decode_video`, seeds `loaded_bgr`) |
| `c975a6e` | collector `--static-frac`, `--no-reset`, `--park-label`, `--align-video` |
| `6b96dfb` | aligner `.aligning` claim-file so concurrent aligners can't race the `.mov` |

**UNCOMMITTED, still in the working tree — commit these:**
- `scripts/data/collect_sls_xctest.py`: `--max-segments N` flag (loop-exit after N segments).
- `scripts/ops/mvp_collect.sh`: the per-segment collection loop (untracked).
- `experiments/vision_sequence_leap_journal.md`: modified (pre-existing, not mine — check before committing).
- `scripts/inspect/review_modal_corpus.py`: untracked (pre-existing from before this session).

Modal: volume `trueskate-mvp` created (empty). Full test suite green (105/105) at every commit.

## Verified vs open

**SOLID — Stage 0 gate passed.** A stationary touch renders the real orange mark at the
commanded point; localisation error median 0.006 normalised; visible duration tracks the
hold (0.2s→0.33, 0.5s→0.67, 1.0s→1.17, 2.0s→2.13; tap→~0.2s ≈ 6 frames @30fps). So onset
AND liftoff are both recoverable. The hub bar (ME/SKATEPARKS/…) is present because the
board is stationary — that is NORMAL, not menu contamination, and is present at inference
too (memory `hub-bar-normal-when-board-stationary`).

**SOLID — video storage.** On the static Workshop scene: crf 20 → 158× smaller, 32× fewer
inodes, mark survives intact to crf 30. (SLS drag corpus benchmarked ~30× — motion costs
more.) Loader equivalence verified: worst mean |PNG−MP4| = 1.83/255.

**BROKEN — timing alignment. This is the blocker.** The aligned samples' `frame_time = 0`
does NOT reliably land on the touch's first pixels. Measured command→pixel latency DRIFTED
from ~1.1s (morning) to ~2.35s (evening), with within-segment std up to 0.51s. Four
hypotheses raised and ALL FALSIFIED:
1. "Constant Δ is valid" — the 4-recording delta test ran inside ~1 min, too short to see drift.
2. "Anchor degrades within a process" — three fresh processes gave +1.3–1.47s, worse than a multi-segment run's segment 0.
3. "XCTest attachment backlog" — cleared 70 attachments, no improvement.
4. "Stale Appium/WDA (5d22h uptime)" — restarting to test this is what broke the rig; never got the measurement.
Onset error correlates with NONE of: hold duration (+0.07), call_wall (+0.02), gesture
index (−0.38, weak), tap-vs-hold. Root cause UNKNOWN.

⚠️ Stage 0's "0.000 residual" was CIRCULAR: `_DELTA_ACTIONCHAINS_S` was refined 0.98→1.06
using those same 9 holds. Treat all absolute Δ numbers in `align_xctest_traces.py` as
provisional — they were fit to a moving target and the target has since moved ~2×.

## Recommended next step

**Per-segment self-calibration** is the only alignment approach that assumes nothing
already falsified. Each segment already fires ~20% taps at KNOWN positions; detect each
tap's mark by frame-differencing at its commanded (x,y), solve one time-offset per segment,
apply it to that segment's `frame_times`. Absorbs the drift whatever its cause; needs no
rig change; generalises to SLS via periodic injected calibration taps.
- Circularity caveat (narrow, acceptable): timing is derived from the rendered mark, but
  POSITIONS still come from the command manifest — no positional leakage. A systematic
  detector bias would bake into timing labels; the frame-diff detector has been reliable
  in every non-drifting test, so this is low-risk but should be stated, not hidden.
- Before committing hours of collection: run ONE calibrated segment and re-verify with
  `tmp/verify_mvp.py` (expect onset median |err| ≤ ~0.10s). Do not scale until it passes.

Then: Stage 5 (train). `train_trace_extractor_modal.py` already filters by park via
`data_match` — point `corpus` at `trueskate-mvp`, `data_match="workshop"`, train FROM
SCRATCH (a warm start from SLS checkpoints confounds the diagnostic). **Set `latency_s=0`**:
Δ is already applied in the aligner, so the legacy 0.2 would double-count.

## Tooling built this session (rig `tmp/`, and my scratchpad)

All on the rig under `/Users/training-server/trueskate-ai/tmp/` (mirrored from my local
scratchpad; re-copy from repo history if gone):
- `stage0_hold_probe.py` / `stage0_analyze.py` — fire holds+taps, frame-diff for onset/localisation.
- `delta_stability.py` — Δ across N recordings (⚠️ ran too briefly; extend session length before trusting).
- `verify_mvp.py` — ground-truth check: decode sample, frame-diff at labelled point, report onset vs `frame_time 0`. **The go/no-go gate for alignment.**
- `by_seg.py` / `corr.py` — group onset error by segment / correlate with gesture props.
- `verify_anchor.py`, `verify_video.py`, `crf_sweep.py` — aligner/storage validators.

The `frame-difference-at-commanded-point` primitive is the reusable core: static scene +
pre-touch reference median → residual blob = rendered mark, model-independent.

## Corpus side note (deferred, not this MVP)

The main `trueskate-corpus` Modal volume is at **997.9/1024 GB and 2.37M files vs a 500k
inode limit** — effectively full on both axes. ~48% of sampled frames are park-editor
contamination. The original `align_xctest_traces.py` bug (anchored on `t_call_end`, the
Appium RETURN time, median 1.7s after the touch) means ~43% of corpus `frame_000`s already
show a fully-drawn trace. No deletion was done (Asher: "don't delete anything"). A dry-run
classifier (`scratchpad/classify_corpus.py`) exists but TIMED OUT at 7200s/container — if
resumed, shard per-park not per-session and probe fewer frames/sample.

## Memories written/updated this session

`sls-window-anchored-to-call-end`, `xctest-command-to-pixel-delta` (updated: Δ NOT constant
under load / drifts), `hub-bar-normal-when-board-stationary`, `wda-signing-free-team-7day-expiry`
(updated: never restart a running WDA to test something).
