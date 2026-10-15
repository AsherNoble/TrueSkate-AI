# Trick-speed batch: curved Model 1 data at real trick speeds

Approved by the operator on 2026-10-15 (explore mode). Experiment ID
`TRICK-SPEED-20261015`. Code: `src/trueskate_ai/model1/trick/`; command line:
`scripts/collection/trick_speed_batch.py`.

## Question and link

Can WDA record clean, correctly labelled strokes at the operator's real trick
speeds in Workshop, so a curved Model 1 can learn to read the filmed kickflip?
This addresses the speed gap (SYSTEM_MAP blocker 1) and feeds step 4, the rough
curved MVP.

## What is recorded

- XR1 in Workshop at the kickflip waypoint (saved flatground, clear floor facing
  the closed door), in one-minute 60 fps recordings.
- **Strokes:** 50–600 ms, drawn log-uniformly (about two thirds are under
  200 ms), with 1–4 straight pieces of at least 40 ms each. Some start with a
  short hold.
- **Stroke families:** push, pop, flick and catch, jittered around the approved
  `candidate_19` kickflip, plus random strokes anywhere in the safe area.
- **Runs:** each clip holds a run of 1–4 strokes.
  - Half the runs follow the kickflip's order (push → pop → flick → catch, or
    part of it); half mix families.
  - Each stroke is its own request to WDA's direct endpoint, sent when the
    previous request returns plus 0–150 ms.
  - A run must fit one 120-frame clip, with five frames either side.
- **Each minute:** die-five markers at 1.5, 29 and 57 s and resets at 26 and
  53.5 s, as timing-instrumented W3C requests. About 15–20 runs fit between them.
- **Seeds:** run *n* is drawn from seed `[20261015, n]`. A run too long for one
  clip is skipped, never sent.

## Labels and checks

- **Labels** are the commands sent: quantized points and piece times.
  - A stroke's start time is the rig-clock call start plus the recording's
    median rig-to-WDA offset, because direct requests carry no WDA timing record.
  - The markers give the video clock; the held-out middle marker must fall
    within two frames.
- **No screenshot menu checks** (operator decision 2026-10-15; see
  `DEPLOYMENT.md`).
- **Kept:**
  - the foreground check before every run;
  - the protected-control map on every path;
  - calibration, native frame timestamps and decoded frame counts;
  - the recorder failure caps and the rig busy lock.
- **Where the work runs:** recording happens on the rig; clip cutting and
  checks run unattended on the laptop.
- **Spot check:** the operator rates 20 clips as OK, Fault or Unsure, looking
  only for gross faults (missing or extra touches, menus). A ring drawn on each
  clip marks the commanded position. The page is the curved audit review UI;
  Export saves the report in the run's `review/` folder and stops the page's
  server, and the report is imported with `verdicts --report`.

## Box and decision

- **Size:** a 100-clip trial (about 30 min of XR1), then about 1,000 clips
  (about 1–2 h of XR1, mostly unattended).
- **Deadline:** before XR1's WDA signing expires (2026-10-21 00:57 UTC).
- **Cost:** no spend. Training is a separate proposal (step 4).
- **Decision:** if the trial keeps at least 70% of its clips and the spot check
  finds no gross faults, record the 1,000 and propose step 4. Otherwise, fix the
  cause and rerun the 100-clip trial.
- **Known limits:**
  - WDA cannot start a stroke sooner than about 0.25–0.3 s after the previous
    one ends; real play has a 0.13 s pop-to-flick gap.
  - Direct-request timing is an estimate; the spot check's ring shows gross
    offsets.
  - Curved v1 marks one touch per clip, so reading runs needs a model change in
    step 4.
