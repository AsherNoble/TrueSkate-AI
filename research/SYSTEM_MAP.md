# System map

Updated 2026-10-16. Read this page first: what we're building, how the parts
connect, what's blocking and the decision in front of you. Numbers link to their
records; `[unsure]` marks the unchecked; unmerged work is named in plain text.

**North star: by 2026-11-20, a working version of the whole project, even a
crap one** (set by you on 2026-10-16). Model 2 plays True Skate live, and the
intended trick lands through WDA or the robotic hand.

## The loop

1. **Expert video** of you playing.
2. **Model 1**, offline, labels the touch gestures in that video.
3. **Model 2** learns from those labels (behaviour cloning), then plays live:
   recent screen frames plus its last gestures in, next gestures out.
4. An **executor** plays the gestures on the phone: WebDriverAgent (WDA)
   today. The robotic hand takes the same gestures through a WDA-shaped API; it
   is being built in parallel in the hardware repo.
5. **Outcome:** did the intended trick land?

Model 1 learns from self-labelled data: the executor plays random gestures, the
phone records the video, and the known gesture becomes the label.

## Clock and latency chain

| Link | Size | Solved? |
|---|---|---|
| WDA command → touch visible on video | 118–174 ms delay ([gate](experiments/M1-TIMING-GATE-20260926.md)) | Yes: calibration taps in every one-minute recording measure it |
| Phone clock vs video clock | drifts about −1,900 ppm, one frame off after ~23 s ([ALIGN](experiments/ALIGN-20260913-wda-onset.md)) | Yes: start and end anchors; held-out middle checks land within ~6–25 ms ([length](experiments/LINEAR-LENGTH-20261003.md)) |
| Gesture label time vs the visible trail | direct trick gestures timed by the rig clock sat −23 to +25 ms off their trail (10–90%), mostly a per-recording offset; the pilot's phone-timestamped touches sat −16 to +4 ms (middle half). No clear speed error once the trail's glow is allowed for ([journal](JOURNAL.md)) | Yes for trick gestures: admission re-times each gesture to its own trail. Speed-ups seen after bends, and pops whose trail stops ~10–20 pt short (about a quarter of pops), remain |
| Back-to-back WDA gestures | the next starts at least ~0.25–0.3 s after the last ends, with ~100 ms jitter ([kickflip](experiments/KICKFLIP-REPEATABILITY-20261009.md)) | No, it's an XCTest limit. The Pico avoids it, and the kickflip lands anyway |
| Pico USB mouse → touch | ≤32 µs board lateness at 3 ms reports; 1 ms unvalidated ([HID-USB](experiments/HID-USB-20261006.md)) | Timing yes; visible contact accuracy unproven |
| Video capture | 30 fps for collection (33 ms per frame); 60 fps for diagnostics and curved v1 | Choice made |
| Live frames for Model 2 (WDA's MJPEG stream) | tops out at 27–29 new frames a second whether asked for 30 or 60 fps, at half (414×896) or quarter (207×448) size; none stale (0 of 374 spans where the screen moved), and the 60 fps recording runs unaffected. A tap shows in the stream about as soon as in the recording, 0.11–0.16 s after WDA sends it ([journal](JOURNAL.md)) | Yes: ask for 30 fps at 25%; the live reader always takes the newest frame and drops any backlog |
| What linear Model 1 sees | 32 frames over 2.3 s: one frame every ~73 ms | Too sparse for trick gestures ([check](JOURNAL.md)); tricks use every 60 Hz frame, as curved v1 does |
| Shortest gesture the game counts | the board moves at ≥15 ms, not at 10 ms ([length](experiments/LINEAR-LENGTH-20261003.md)) | Known |
| Finger trail on screen | fades to half in ~92 ms, gone by ~194 ms (branch record EXEC-TRANSFER-20261013); on your filmed kickflip, visible by eye for ~130–150 ms | Known, rough |
| **Kickflip outcome tolerance** | timing: landed 19/19 with pop→flick starts 322–429 ms apart (you: ~205 ms) and ~100 ms jitter ([kickflip](experiments/KICKFLIP-REPEATABILITY-20261009.md)). Position, with the pop and flick moved a fixed distance: landed 10/10 at 0, 8/10 at 2% of the screen, 1/9 at 5%, 0/10 at 10% ([tolerance](experiments/KICKFLIP-TOLERANCE-20261015.md)) | Timing tolerance is wide. **Position must be within ~2% of the screen**: the bar for labels, Model 1 and the hand |

## Models: what goes in, what comes out

- **Linear Model 1** (trained, current).
  - In: 32 colour frames at 64×144 spanning 2.3 s, with one linear gesture
    starting at a fixed frame.
  - Out: start point, end point and duration.
  - Training data: linear gestures of 0.30–1.20 s.
  - Results: 91.04% of unseen test gestures within 0.03 of the screen and 0.10 s
    ([HALFRES](experiments/M1-HALFRES-20260928.md)); 87.32% in unseen parks
    ([EXPAND](experiments/M1-EXPAND-20260929.md)). Misses are mostly short end
    points on fast gestures ([DIAG](experiments/M1-DIAG-20260928.md)).
- **Curved Model 1 v1** (built, never trained): 120 frames at 60 Hz in, up to
  10 segments over 0.3–1.2 s out. Its data check scored 83/100 against a
  95/100 gate, so no training data was collected.
- **Trick Model 1 MVP v0** (trained 2026-10-16, rough): curved v1's network,
  120 frames at 128×288 in, up to 4 gestures of 50–600 ms out, each 1–4 segments
  of at least 40 ms. On 78 test gestures: 99% on time at both ends; 90%
  also within 5% of the screen everywhere, but only 37% within the kickflip's
  2% ([record](experiments/MODEL1-TRICK-MVP-20261016.md)).
- **Model 2** (September code, untrained; critical path for 2026-11-20).
  - In: recent frames at 30 fps, quarter size (207×448), plus your last gestures.
    Out: the next gestures.
  - Trained on your play, labelled by Model 1 (`scripts/model2/`). The code
    still uses the old 9-number gesture and a retired executor path, so it
    needs adapting to timed multi-segment gestures and today's WDA path.

## Executors

- **WDA** (current): one request per gesture, a hard rule in
  [GESTURES.md](../GESTURES.md). It delivered your 75 ms pop and 60 + 40 ms
  flick well enough to land 19/19.
- **Pico USB mouse** (pilot): the phone treats it as a pointer. Its timing is
  precise and it has no gap limit, but contact accuracy isn't yet shown
  ([HID-USB](experiments/HID-USB-20261006.md)).
- **Robotic hand** (in parallel, in the hardware repo): it takes the same
  gestures through a WDA-shaped API, so Model 2 never knows which executor
  played. Its accuracy spec is ~2% of the screen (the outcome tolerance above).

## Data and rig

- Linear corpus: `model1_linear_good13100_20260927`, plus a doubling to 18,394
  training clips. Timing checks screened it, and blind 100-clip audits passed it
  ([audit](experiments/M1-CORPUS-AUDIT-20260927.md)).
- Collection is OFF unless you authorise a bounded run. Rig release `189dfde`.
- The XR1 learn-replay pilot finished on 2026-10-15 (branch
  `codex/learn-replay-pilot-20261013`): 430 curved clips at 0.3–1.2 s; its curved
  Model 1 passed 25% of 80 test clips; replay skipped. Its session owns the record.
- Trick-speed trials: 125 clips each on XR1 and XR2 (TRICK-SPEED-20261015),
  with labels re-timed to the visible trail ([journal](JOURNAL.md)).
- Training runs on Modal. Three seeds of linear Model 1 cost about $12.5 at
  list prices ([COLAB](experiments/M1-COLAB-20261006.md)).

## What's blocking the end-to-end loop

1. **Speed gap: trick-speed data is only in trial size.** Your kickflip's pop
   takes 75 ms and its flick 100 ms
   ([kickflip](experiments/KICKFLIP-REPEATABILITY-20261009.md)). Both
   Model 1 training sets start at 0.30 s. Trick models read every 60 Hz
   frame. The two 125-clip trials are in; the 1,000-clip batch is next.
2. **Trick Model 1 is rough.** MVP v0 finds the gestures and times them, but
   places them within the kickflip's 2% only 37% of the time (90% at 5%).
3. **Model 2 is untrained.** Its live feed is measured: ~28 fps at quarter size.
4. **Ground truth from your own play (parked).** A stock iPhone can't log
   touches; an Android phone is a last resort (2026-10-15).

## Open decisions

- **Curved v1's 95/100 data gate:** keep it, change it or drop it. You're
  deciding this in a separate session; no recommendation here.

## Next experiments, ranked (approved 2026-10-16; a living list)

Add or reorder items at any time; each comes to you as a short proposal first.
Rough calendar: 1–3 by about Oct 24, 4–5 by Oct 31, 7 by Nov 10, 8 by Nov 17.

1. **Kickflip position tolerance: done 2026-10-16** ([record](experiments/KICKFLIP-TOLERANCE-20261015.md)):
   lands 8/10 at 2%, 1/9 at 5%, so the bar is ~2% and item 10 comes first.
2. **Trick-speed 1,000 clips on XR1 + XR2** ([protocol](protocols/trick_speed_batch.md)),
   after item 10 and with its pop fix. You call the remaining faults minor →
   record; else → fix the cause and rerun a 100-clip trial.
3. **Record your own play** in Workshop (30–60 min, the rig's 60 fps
   recorder) as Model 2's data. The trail shows clearly → keep it; else →
   change the settings and record again.
4. **Rough curved Model 1 MVP v0: done 2026-10-16** ([record](experiments/MODEL1-TRICK-MVP-20261016.md)):
   90% of test gestures pass at 5%, 37% at 2%. Item 1 says 2%, so fix the gesture
   ends first (pops' fall ~13 pt short; item 10), then retrain on the 1,000.
5. **Replay check of Model 1's labels:** your filmed kickflip, replayed by
   WDA, 20 times. Lands in at least half → Model 1 can label Model 2's data;
   else → fix the weakest link.
6. **Live frames for Model 2: done.** WDA tops out at ~28 fps; live play asks
   for 30 fps at 25% size, and Model 2 trains on 30 fps frames at 207×448.
7. **Model 2 v0 on your labelled play** (Modal under $10). Sensible offline
   predictions → item 8; else → more play, or a simpler "kickflip now / wait"
   target.
8. **Model 2 plays live** through WDA, with a hand-ready interface. It lands a
   kickflip → north star met; else → fix the failing link.
9. Speed after a bend, only if items 1, 5 or 8 point to it: re-time each segment.
10. **Pops' missing end (next):** a 16–33 ms pause at the end, ~15 min of XR1. The trail
    shows the whole pop → add the pause; else → label pops' ends from the trail.
11. Timing tolerance inside a gesture, only if item 5 fails with good positions.

## Parked

- The Bluetooth mouse: a 15 ms radio interval and 3–5% late reports
  ([HID-POINTER](experiments/HID-POINTER-20261004.md)).
- Multi-finger and spin gestures.
- More linear scaling: data still helps, but the kickflip slice doesn't need it.
- Replay fine-tuning, after the MVP ([note](protocols/model1_replay_finetuning.md)).
- WDA-side timestamps for direct gestures (needs a WDA rebuild); Pico touch
  accuracy; other parks, until the Workshop kickflip works.
