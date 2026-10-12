# Label tolerance: continuation, 2026-10-13

**Finding:** an inverse model can absorb an invertible deterministic executor warp for **same-executor replay**. On human video its target is then the executor command reproducing that touch, not necessarily the human's original contact path. Determinism alone neither guarantees learnability nor sets a universal timing tolerance. Non-injective mappings prevent unique original-command recovery; a suitable canonical command can still reproduce the effect.

**Primary literature:** [VPT](https://cdn.openai.com/vpt/Paper.pdf) uses tick-synchronized recording and128-frame noncausal inference; its20Hz interface is not a±50ms label-noise allowance. [LAPO §6.4](https://arxiv.org/pdf/2312.10812) explicitly warns that latent actions track delayed visible effects and recommends more past/future context. [Temporally correlated noise](https://proceedings.mlr.press/v162/swamy22a/swamy22a.pdf) can confound ordinary BC. [RTC](https://papers.nips.cc/paper_files/paper/2025/file/300ccb2187dedd4edcc07f7e76d8e553-Paper-Conference.pdf)'s+100/+200ms robustness concerns compensated inference delay, not corrupt labels. No reviewed paper establishes a transferable±25ms threshold. More primary sources and mechanism distinctions: [literature notes](continuation-20261013/LITERATURE.md).

**CPU experiment:**15 fits:3 independent data/model seeds×5 conditions; each2400 train/600 held-out clips,60Hz,72 frames,16×16 blob/trail videos, pixel-derived intensity moments→MLP→command timings. Known fixed bent path makes this a deliberately favorable timing-only test. Deterministic warp includes25ms lag,segment rate changes and an invertible move/hold redistribution; noise adds5ms onset SD/3% duration SD. Evaluate continuous reconstructed commands and fresh independently noisy replays, with no decoder confound. Runtime54s.

| Condition | Command XY MAE | Fresh replay XY MAE | Fresh replay pass %, mean±seed SD |
|---|---:|---:|---:|
| Identity | .00337 | .00337 |97.9±1.2|
| Deterministic warp | .00452 | .00438 |91.2±2.5|
| Random±25ms | .02719 | .03044 |31.6±1.3|
| Warp+noise | .01455 | .01549 |31.4±1.6|
| Non-injective collapse | .01799 | .00471 |69.4±5.0|

XY MAE is normalized Euclidean error on contact overlap. Pass requires95% of union-contact samples active in both and within.03, plus onset/release within2/60s. These are toy outcomes, not rig performance. Deterministic raw command/trace error is≈.055 despite good inverse replay; on the same videos interpreted as human raw-touch targets, unexecuted predicted commands retain≈.055 bias. Identical collapse videos hide120ms move/hold differences, proving≥60ms balanced-pair per-duration RMSE. Known-executed-time random-onset oracle still has11.07ms MAE; learned14.47ms also includes model error.

**Qualification:** retain measurable outcome bounds; compare observations against a held-out forward transfer model and test fresh executor repeatability, supported human coverage, actual model/decoder capacity, and stationarity across device/build/park. Declare command versus raw-touch label semantics and measurement uncertainty. For straight movement at speed v, timing error δ gives≈v|δ| XY error; .03 implies≈.03/v seconds, not one tolerance for all strokes. Global injectivity is needed for unique command recovery, but effect imitation can instead use a validated right inverse. Existing8ms/one-frame proposal remains an unvalidated engineering gate.

**Confidence:** high on mathematical distinctions and exact collision checks; moderate on this reproducible favorable toy; low on actual executor/human transfer until measured. Hold collapse is hypothetical: the sibling trace audit invalidated clip67's apparent170ms lead. Preserved original code had no completed results and confounded nominal noisy replay/decoder/hold thresholds. Full limitations, analytical checks, provenance and reproduction: [methods](continuation-20261013/METHODS.md), [script](continuation-20261013/study.py), [results](continuation-20261013/results.json), [verification](continuation-20261013/verification.json).
