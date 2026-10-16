# System map

Updated 2026-10-16. Read this page first: what we're building, how the pieces
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
| Stroke label time vs the visible trail | direct trick strokes timed by the rig clock sat −23 to +25 ms off their trail (10–90%), mostly a per-recording offset; the pilot's phone-timestamped touches sat −16 to +4 ms (middle half). No clear speed error once the trail's glow is allowed for ([journal](JOURNAL.md)) | Yes for trick strokes: admission re-times each stroke to its own trail. Speed-ups seen after kinks, and pops whose trail stops ~10–20 pt short (about a quarter of pops), remain |
| Back-to-back WDA gestures | the next starts at least ~0.25–0.3 s after the last ends, with ~100 ms jitter ([kickflip](experiments/KICKFLIP-REPEATABILITY-20261009.md)) | No, it's an XCTest limit. The Pico avoids it, and the kickflip lands anyway |
| Pico USB mouse → touch | ≤32 µs board lateness at 3 ms reports; 1 ms unvalidated ([HID-USB](experiments/HID-USB-20261006.md)) | Timing yes; visible contact accuracy unproven |
| Video capture | 30 fps for collection (33 ms per frame); 60 fps for diagnostics and curved v1 | Choice made |
| Live frames for Model 2 (WDA's MJPEG stream) | tops out at 27–29 new frames a second whether asked for 30 or 60 fps, at half (414×896) or quarter (207×448) size; none stale (0 of 374 spans where the screen moved), and the 60 fps recording runs unaffected. A tap shows in the stream about as soon as in the recording, 0.11–0.16 s after WDA sends it ([journal](JOURNAL.md)) | Yes: ask for 30 fps at 25%; the live reader always takes the newest frame and drops any backlog |
| What linear Model 1 sees | 32 frames over 2.3 s: one frame every ~73 ms | Too sparse for trick strokes ([check](JOURNAL.md)); tricks use every 60 Hz frame, as curved v1 does |
| Shortest gesture the game counts | the board moves at ≥15 ms, not at 10 ms ([length](experiments/LINEAR-LENGTH-20261003.md)) | Known |
| Finger trail on screen | fades to half in ~92 ms, gone by ~194 ms (branch record EXEC-TRANSFER-20261013); on your filmed kickflip, visible by eye for ~130–150 ms | Known, rough |
| **Kickflip outcome tolerance** | landed 19/19 with pop→flick starts 322–429 ms apart (you: ~205 ms) and ~100 ms jitter ([kickflip](experiments/KICKFLIP-REPEATABILITY-20261009.md)) | Timing tolerance is wide. **Position tolerance is unknown; measure it next** |

## Models: what goes in, what comes out

- **Linear Model 1** (trained, current).
  - In: 32 colour frames at 64×144 spanning 2.3 s, with one straight swipe
    starting at a fixed frame.
  - Out: start point, end point and duration.
  - Training data: swipes of 0.30–1.20 s.
  - Results: 91.04% of unseen test swipes within 0.03 of the screen and 0.10 s
    ([HALFRES](experiments/M1-HALFRES-20260928.md)); 87.32% in unseen parks
    ([EXPAND](experiments/M1-EXPAND-20260929.md)). Misses are mostly short end
    points on fast swipes ([DIAG](experiments/M1-DIAG-20260928.md)).
- **Curved Model 1 v1** (built, never trained; branch `codex/curved-model1-v1`).
  - In: 120 frames at 60 Hz (2 s).
  - Out: a timed path of up to 10 straight pieces, 0.3–1.2 s long, each piece
    at least 100 ms.
  - Its data check scored 83/100 against a 95/100 gate, so no training data
    was collected.
- **Model 2** (September code, untrained; critical path for 2026-11-20).
  - In: recent frames at 30 fps, quarter size (207×448), plus your last strokes.
    Out: the next strokes.
  - Trained on your play, labelled by Model 1 (`scripts/model2/`). The code
    still uses the old 9-number stroke and a retired executor path, so it
    needs adapting to timed multi-piece strokes and today's WDA path.

## Executors

- **WDA** (current): one request per gesture, a hard rule in
  [GESTURES.md](../GESTURES.md). It delivered your 75 ms pop and 60 + 40 ms
  flick well enough to land 19/19.
- **Pico USB mouse** (pilot): the phone treats it as a pointer. Its timing is
  precise and it has no gap limit, but contact accuracy isn't yet shown
  ([HID-USB](experiments/HID-USB-20261006.md)).
- **Robotic hand** (in parallel, in the hardware repo): it takes the same
  gestures through a WDA-shaped API, so Model 2 never knows which executor
  played. Its accuracy spec is the outcome tolerance above.

## Data and rig

- Linear corpus: `model1_linear_good13100_20260927`, plus a doubling to 18,394
  training clips. Timing checks screened it, and blind 100-clip audits passed it
  ([audit](experiments/M1-CORPUS-AUDIT-20260927.md)).
- Collection is OFF unless you authorise a bounded run. Rig release `189dfde`.
- The XR1 learn-replay pilot finished on 2026-10-15 (branch
  `codex/learn-replay-pilot-20261013`). It collected 430 curved clips at
  0.3–1.2 s and trained a curved Model 1 on 320, which passed 25% of 80 test
  clips; replay was skipped. Its own session owns the record.
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
2. **No curved Model 1 that reads fast strokes.** Curved v1 stopped at its data
   gate (83/100 against 95). The pilot's model passed 25% at 0.3–1.2 s.
3. **Model 2 is untrained.** Its live feed is measured: ~28 fps at quarter size.
4. **Ground truth from your own play (parked).** A stock iPhone can't log
   touches; an Android phone is a last resort (2026-10-15).

## Open decisions

- **Curved v1's 95/100 data gate:** keep it, change it or drop it. You're
  deciding this in a separate session; no recommendation here.

## Next experiments, ranked (approved 2026-10-16; a living list)

Add or reorder items at any time; each comes to you as a short proposal first.
Rough calendar: 1–3 by about Oct 24, 4–5 by Oct 31, 7 by Nov 10, 8 by Nov 17.

1. **Kickflip position tolerance:** review the 39 recorded attempts. Landings
   hold at 5% noise → the current labels are good enough; else → item 10 first.
2. **Trick-speed 1,000 clips on XR1 + XR2** ([protocol](protocols/trick_speed_batch.md)).
   The XR1 trial's trail check finds nothing new and you call the remaining
   faults minor → record; else → fix the cause and rerun a 100-clip trial.
3. **Record your own play** in Workshop (30–60 min, the rig's 60 fps
   recorder) as Model 2's data. The trail shows clearly → keep it; else →
   change the settings and record again.
4. **Rough curved Model 1 MVP v0:** up to 4 strokes per clip, pieces down to
   40 ms, Modal under $5 ([MVP test](protocols/model1_curved_mvp2.md)). At
   least half the validation strokes within item 1's tolerance → items 5 and 7;
   else → sort failures into timing vs path.
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
9. Speed after a kink, only if items 1, 5 or 8 point to it: re-time each piece.
10. Pops' missing end, only if item 1 says it matters: test a 16–33 ms end hold.
11. Timing tolerance inside a stroke, only if item 5 fails with good positions.

## Parked

- The Bluetooth mouse: a 15 ms radio interval and 3–5% late reports
  ([HID-POINTER](experiments/HID-POINTER-20261004.md)).
- Multi-finger and spin gestures.
- More linear scaling: data still helps, but the kickflip slice doesn't need it.
- Replay fine-tuning ([note](protocols/model1_replay_finetuning.md)), which
  comes after the MVP.
- WDA-side timestamps for direct strokes (needs a WDA rebuild); Pico contact
  accuracy; other parks, until the Workshop kickflip works.
