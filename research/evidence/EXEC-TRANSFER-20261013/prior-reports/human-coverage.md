# 4 · Human coverage: does the curved MVP sampler cover human True Skate gestures?

Saved by main from the agent's hand-back (subagents cannot write report .md files). The scripts, data and figure are in this folder: `coverage.py`, `extra.py`, `coverage-summary.json`, `extra-summary.json`, `representability-tol003.txt` and `coverage.png`.

## Answer

No, except for slow contacts. 7 of the 8 expert strokes fall outside the sampler's support:

- The 5 pushes and 2 flicks last 57–115 ms, but the sampler starts at 300 ms.
- They average 3,700–5,100 pt/s, a speed only 0.7% of sampler draws reach.
- Only the 550 ms catch is inside.

These limits are structural. The contract sets `MIN_DURATION_MS` 300 and `MIN_SEGMENT_MS` 100, and the decoder only searches contacts of 300–1200 ms. So Model 1's label for a push or flick is extrapolation by construction. **Confidence: high.**

## Method

- **Sampler:** `sample_gesture`, 5,000 seeds each of `random_chain` and `arc`, run read-only with `PYTHONPATH=src`. Lengths are in 414×896 points.
- **Human evidence:** the XR2 Workshop 360-flip demo, using both extractions (the conclusions hold for either):
  - v2 centroid: `research/evidence/CURVE-AUDIT-20261003/demo-replay/extracted-gestures.json`
  - v3 leading edge: `research/evidence/HID-POINTER-20261004/demo-replay-v3/strokes-v3.json`
- **Secondary sources:** the hand-guessed kickflip, the trick-library recipes (found by CMA-ES, not human), web sources and the MicroJam swipe study.

## Coverage gaps

| Dimension | Sampler | Human evidence | Gap |
|---|---|---|---|
| Stroke duration | 300–1200 ms, uniform | Pushes 89–115 ms, flicks 57–67 ms, catch 550 ms (demo). Kickflip hand guess: 50–60 ms. Trick libraries: 44.5% under 300 ms. Phone swipes: median 130 ms (MicroJam). | **Critical**: 0% coverage below 300 ms |
| Segment duration | At least 100 ms (median 138 / 152 ms) | Within the 0.03 tolerance, a push or flick needs 2–3 segments of 17–69 ms. | **Critical** |
| Mean speed | Median 992 / 180 pt/s; 0.7% reach 3,000 pt/s; maximum 4,973 pt/s | 3,700–5,100 pt/s for pushes and flicks; 337 pt/s for the catch | **Critical** |
| Peak speed | 95th percentile 4,517 / 664 pt/s | Per frame 4,400–7,300 pt/s; peak/mean 1.15–1.54 | **High** |
| Duration, length and speed together | Push-like draws: 16 of 10,000. Flick-like draws: 0 of 10,000. | Path lengths of 185–527 pt are covered alone, but not combined with the duration and speed. | **Critical** |
| Curvature and turns | Arc total turn 0–160°. Chains: median vertex turn 137°; 30% turn 180° or more in total. | Fast strokes are gentle arcs: sagitta/chord 0.02–0.08, total turn 20–50°. | **Medium**: about half the budget goes to zigzags, which the evidence never shows |
| Holds | 15.7% of chains have one exactly still hold of 100–946 ms; arcs have none. | The catch is a slow drift, not a still hold. Grinds, manuals and spins involve held or corrected contacts, often over 1.2 s. | **Medium** |
| Start/end regions | Safe rectangles weighted by area | All 16 demo endpoints lie inside them. Push ends at y 0.82–0.84 sit close to the excluded band above 0.88. | **Low** (one park) |
| Gaps between strokes | One contact per clip | Gaps of 118–650 ms; 4–7 contacts per 2 s window; only the catch can be cut out as a single-contact clip. | **Critical** for real gameplay; out of MVP scope by design |
| Multiple fingers | Single contact only | Official text: "2 fingers / 2 thumbs". Spin needs a second finger. 11% of trick-library delays are negative (contacts overlap). | **High**; out of MVP scope by design |

## What this means

- **Decoder:** given a push, it stretches it by at least 200 ms (12 frames or more, against a 2-frame tolerance) or joins several pushes across the gaps where the finger is lifted.
- **Frozen human clips:** the ten clips can only be catches, slow drags or holds, so they won't test labelling of pushes and flicks.

## Suggested sampler changes, ranked by impact

1. **Add a fast-stroke family, about 30–40% of draws.**
   - Duration 40–300 ms (log-uniform), 1–3 segments of at least 33 ms each, length 100–600 pt, slow-fast-slow timing, speeds up to about 8,000 pt/s.
   - `MIN_DURATION_MS`, `MIN_SEGMENT_MS` and the decoder interval must change together.
   - Executor evidence: WDA drew 75/75 single 30–50 ms moves (LINEAR-LENGTH), and points at least 33 ms apart were 76–91% Good (CURVE-AUDIT). Re-qualify execution at these speeds before collecting.
2. **Add clips with 2–6 sequential contacts, 100–700 ms apart.** This needs multi-contact labels and decoding. XCTest needs at least about 0.25–0.33 s between contacts, so the demo's shorter gaps need the Pico.
3. **Rebalance shapes:** keep chain turns mostly under 45°, add fast gentle arcs, cut zigzags to about 10%.
4. **Replace exactly still holds with slow drift** (30–300 pt/s with wobble), and add a catch shape (plant, then move).
5. **Later:** contacts over 1.2 s and variable clip length; a second simultaneous finger; paths across safe rectangles.

## Confidence

- **High** for duration, speed and segment gaps: structural, and every source agrees.
- **Medium-low** for curvature, holds and regions: 8 strokes from one player, one trick, one park.

## Sources

- [App Store listing](https://apps.apple.com/US/app/id549105915)
- [PocketGamer](https://www.pocketgamer.com/true-skate/three-things-to-know/)
- [TrickDojo](https://trickdojo.com/how-to-manual-true-skate-tutorials/)
- [TouchArcade forum](https://devforums.toucharcade.com/threads/true-skate-by-true-axis.161273/page-54) (search summary only)
- [MicroJam](https://arxiv.org/pdf/1902.00680)
