# DECODER-NOISE-20261013 — fixed-knot refit under prediction stress

Offline synthetic experiment, on isolated branch `codex/decoder-noise-20261013`
from `dc159db075f60336ba2543fc7e607c00f5188c51`. No model training, production source change,
corpus access, simulator, phone, rig, paid compute or collection was performed.

The prior [decoder representability study](../evidence/EXEC-TRANSFER-20261013/decoder-followup/REPORT.md)
found that refitting internal coordinates at the frozen decoder's chosen knot
times removed five perfect-target audit failures. This independent experiment
tests whether that minimal change survives noisy inputs and boundary-class errors.

The frozen synthetic challenge has 192 legal commands from eight shape and
control-boundary families. Four predeclared corruption laws at six levels,
plus clean coordinates, are crossed with nine independent onset/release class
patterns: 43,200 paired cases. Both variants see identical float32 predictions
and logits. Scoring, quantization, timing restrictions and protected-control
checks remain unchanged. No invalid candidate is discarded or replaced.

The refit improves guarded passes from **15,992 to 17,127/43,200**: 1,467 gains,
332 regressions. It helps independent coordinate noise substantially, but
offers little protection from correlated noise, drift or biased endpoints.
Protected-control compiler rejections increase from 10,884 to 10,931; 440 paths
become unsafe while 393 become safe. These deliberately concentrated synthetic
conditions do not estimate real-model accuracy or incident rates.

Clean coordinates and exact boundary classes score 184→189/192, with five
gains and no regressions. Three alternating-turn cases still fail the frozen
joint score. One-frame class errors remain costly even with exact XY.

Decision: retain the refit as a research candidate for a guarded sequential
pilot; do not promote it to production. Any pilot must count safety rejections
and paired regressions, retain the original guards, and independently measure
actual prediction errors. This study establishes neither trained-model benefit
nor physical executor fidelity.

The [report](../evidence/DECODER-NOISE-20261013/REPORT.md),
[frozen plan](../evidence/DECODER-NOISE-20261013/PLAN-FROZEN.md), executable study,
full paired ledger, seed-fixed fixtures, provenance and verification are in
[the evidence directory](../evidence/DECODER-NOISE-20261013/).
