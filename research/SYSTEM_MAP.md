# System map

Updated 2026-10-14. Read this page first. It covers what we're building, how the
pieces connect, what's blocking, and the decision in front of you. Numbers link
to their records, and `[unsure]` marks anything not yet checked. Work that
exists only on an unmerged branch is named in plain text. This page replaced
`STATUS.md`; its history lives in the records and the journal.

## The loop

1. **Expert video** of you landing a trick.
2. **Model 1** reads the touch gestures out of the video.
3. An **executor** replays those gestures on the phone: WebDriverAgent (WDA)
   today; a Pico USB mouse (pilot); a robotic hand later `[unsure: tracked
   outside this repo]`.
4. **Outcome:** did the intended trick land? This is the north star.

Model 1 learns from self-labelled data: the executor plays random gestures, the
phone records the video, and the known gesture becomes the label. Model 2
(later) would choose the next gesture itself from gameplay.

## Clock and latency chain

| Link | Size | Solved? |
|---|---|---|
| WDA command → touch visible on video | 118–174 ms delay ([gate](experiments/M1-TIMING-GATE-20260926.md)) | Yes: calibration taps in every one-minute recording measure it |
| Phone clock vs video clock | drifts about −1,900 ppm, one frame off after ~23 s ([ALIGN](experiments/ALIGN-20260913-wda-onset.md)) | Yes: start and end anchors; held-out middle checks land within ~6–25 ms ([length](experiments/LINEAR-LENGTH-20261003.md)) |
| Back-to-back WDA gestures | the next starts at least ~0.25–0.3 s after the last ends, with ~100 ms jitter ([kickflip](experiments/KICKFLIP-REPEATABILITY-20261009.md)) | No, it's an XCTest limit. The Pico avoids it, and the kickflip lands anyway |
| Pico USB mouse → touch | ≤32 µs board lateness at 3 ms reports; 1 ms unvalidated ([HID-USB](experiments/HID-USB-20261006.md)) | Timing yes; visible contact accuracy unproven |
| Video capture | 30 fps for collection (33 ms per frame); 60 fps for diagnostics and curved v1 | Choice made |
| What linear Model 1 sees | 32 frames over 2.3 s: one frame every ~72 ms | **Open**: see blockers |
| Shortest gesture the game counts | the board moves at ≥15 ms, not at 10 ms ([length](experiments/LINEAR-LENGTH-20261003.md)) | Known |
| Finger trail on screen | fades to half in ~92 ms, gone by ~194 ms (branch record EXEC-TRANSFER-20261013) | Known, rough |
| **Kickflip outcome tolerance** | landed 19/19 with pop→flick starts 322–429 ms apart (you: ~205 ms) and ~100 ms jitter ([kickflip](experiments/KICKFLIP-REPEATABILITY-20261009.md)) | Timing tolerance is wide. **Position tolerance is unknown; measure it next** |

## Models: what goes in, what comes out

- **Linear Model 1** (trained, current).
  - In: 32 colour frames at 64×144 spanning 2.3 s, with one straight swipe
    starting at a fixed frame.
  - Out: start point, end point and duration.
  - Training data: swipes of 0.30–1.20 s.
  - Results: the best tested model recovers 91.04% of unseen test swipes, to
    within 0.03 of the screen and 0.10 s
    ([HALFRES](experiments/M1-HALFRES-20260928.md)). Doubling the data to
    18,394 clips raised the held-out score to 91.37%; in parks it never saw
    it scores 87.32% ([EXPAND](experiments/M1-EXPAND-20260929.md)). More data
    still helps.
  - Misses are mostly end points falling short, on fast swipes
    ([DIAG](experiments/M1-DIAG-20260928.md)).
- **Curved Model 1 v1** (built, never trained; branch `codex/curved-model1-v1`).
  - In: 120 frames at 60 Hz (2 s).
  - Out: a timed path of up to 10 straight pieces, 0.3–1.2 s long, each piece
    at least 100 ms.
  - Its data check scored 83/100 against a 95/100 gate, so no training data
    was collected.
- **Model 2** (unfinished).
  - In: recent frames plus your last strokes. Out: the next strokes.
  - It would be trained on your play, labelled by Model 1. It isn't needed for
    the first kickflip.

## Executors

- **WDA** (current): one request per gesture, a hard rule in
  [GESTURES.md](../GESTURES.md). It delivered your 75 ms pop and 60 + 40 ms
  flick well enough to land 19/19.
- **Pico USB mouse** (pilot): the phone treats it as a pointer. Its timing is
  precise and it has no gap limit, but contact accuracy isn't yet shown
  ([HID-USB](experiments/HID-USB-20261006.md)).
- **Robotic hand** (later): its accuracy spec is the outcome tolerance above.

## Data and rig

- Linear corpus: `model1_linear_good13100_20260927`, plus a doubling to 18,394
  training clips. Timing checks screened it, and blind 100-clip audits passed it
  ([audit](experiments/M1-CORPUS-AUDIT-20260927.md)).
- Collection is OFF unless you authorise a bounded run. Rig release `189dfde`.
- Running now: the XR1 learn-replay pilot (branch
  `codex/learn-replay-pilot-20261013`). It is a small curved
  collect–train–replay test at 0.3–1.2 s speeds, capturing until about
  2026-10-15 00:30 UTC. Its own session owns it.
- Training runs on Modal. Three seeds of linear Model 1 cost about $12.5 at
  list prices ([COLAB](experiments/M1-COLAB-20261006.md)).

## What's blocking the end-to-end loop

1. **Speed gap (primary).**
   - Your filmed kickflip's pop takes 75 ms and its flick 100 ms (a 60 ms arc,
     then 40 ms outward) ([kickflip](experiments/KICKFLIP-REPEATABILITY-20261009.md)).
   - Both Model 1 training sets start at 0.30 s, and curved v1 needs every
     piece to last at least 100 ms.
   - Linear Model 1 also sees only one frame every ~72 ms, so a 75 ms stroke
     shows up in one or two frames. The fading trail may help.
   - Curved v1 already reads every 60 Hz frame.
2. **No curved Model 1.** The kickflip's flick is curved, so linear Model 1
   can't express it. Curved v1 stopped at its data gate (83/100 against 95).
3. `[unsure]` **Ground truth from your own play.** It isn't known whether
   touches during real play on a stock iPhone can be captured at all.

## Open decisions

- **Curved v1's 95/100 data gate:** keep it, change it or drop it. You're
  deciding this in a separate session; no recommendation here.

## Next steps

You approved the frame-sampling check (proposal below) on 2026-10-14; it runs
first. After it, my recommended order is below; each step is a separate
proposal for you to approve.

1. The frame-sampling check.
2. A time-boxed check of whether touches from your own play can be captured on
   a stock iPhone. If they can, your play becomes the real test set.
3. Curved data at trick speeds, shaped like your real play: strokes from about
   60 ms and pieces down to about 40 ms, collected as a small XR1 Workshop batch
   (this needs rig time).
4. A rough curved MVP: train it, read your filmed kickflip with it, replay the
   result in Workshop and count landings
   ([MVP test](protocols/model1_curved_mvp2.md)).

**Proposal: frame-sampling check**
- **Question:** at one frame every ~72 ms, how much of a 75–100 ms trick stroke
  can Model 1 see?
- **Link:** video → Model 1. If the strokes aren't visible, no amount of data
  will help.
- **Box:** laptop only, about two hours, $0. It uses existing footage: your
  filmed kickflip at 60 fps and the repeat recordings.
- **Method:** pick frames at Model 1's spacing from every possible starting
  offset. For each stroke, count how many sampled frames show it (trail
  included) and how much of its path they show.
- **If** each stroke shows clearly in at least two sampled frames, sampling is
  fine and the fix is trick-speed data (step 3). **Else** fast-stroke models use
  dense 60 Hz frames, as curved v1 does, and the 32-frame recipe isn't used for
  tricks.

## Parked

- Model 2.
- The Bluetooth mouse: a 15 ms radio interval and 3–5% late reports
  ([HID-POINTER](experiments/HID-POINTER-20261004.md)).
- Multi-finger and spin gestures.
- More linear scaling: data still helps, but the kickflip slice doesn't need it.
- Replay fine-tuning ([note](protocols/model1_replay_finetuning.md)), which
  comes after the MVP.
- Other parks and XR2, until the Workshop kickflip works.
