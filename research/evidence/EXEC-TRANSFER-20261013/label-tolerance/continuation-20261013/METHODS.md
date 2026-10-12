# Experiment methods and audit

Completed2026-10-13 using the repository .venv, PyTorch2.12.0 and NumPy2.4.6 on CPU. No rig, SSH, device or paid/cloud compute. Source read-only; all outputs in this continuation directory. Existing unfinished artifacts remain intact.

## Recovery and audit of stopped work

The stopped session's original assignment is Agent `aef205a0e3a6327d0`, description `label-tolerance research`, created2026-10-12T12:54:28Z in session `e0f1c48e-973a-4ff3-a26a-f5f2350d9a43`. Its preserved `../toy.py` has no completed results/REPORT in the recovered scratch directory. It already proposed command, replay and human metrics. Useful parts were retained conceptually; this continuation is a separate experiment.

Weaknesses to avoid:

1. The original sampler can generate negative residual segment budget: D starts300ms while K can4 and minimum segments total400ms. A purported100ms segment can therefore fall below100ms. No quantitative result from that unfinished script is relied on.
2. Its replay is noise-free nominal even for stochastic executors. This can materially overstate actual repeatability; fresh independent executor noise is required.
3. Decoded paths introduce another error source; holds are classified using eps=.02 instead of the executor's true1e-6. Decode subdivision can change duration warps that are defined per segment. A timing-inverse failure cannot then be attributed uniquely to the learned model. This study uses a fixed command representation.
4. Its seed is fixed; single fits do not show robustness to initialization/data. This study has three fully separate model/data seeds.
5. The old `sampler_tolerance.json` is a **geometric perturbation** analysis, not model evidence. It evaluates positions on the original contact window, clips before/after positions to endpoints and does not penalize contact-mask differences. It cannot support full qualification pass rates. Its random_chain/arc sensitivity contrast is nevertheless plausible: speed matters.

## Generative specification

Command c=(o,d,h,e): onset, first move duration, hold duration, final move duration, in seconds. Independent priors: o∼U(.10,.24), d,e∼U(.14,.30), h∼U(.06,.18). Known fixed bent geometry: (.20,.25)→(.75,.25)→(.75,.75), hold at the kink. This intentionally favorable **timing-only** experiment does not estimate arbitrary True Skate paths.

Executor conditions:

- clean: identity.
- det: +25ms onset; scale first move by1.2 and final move by.8. With D=1.2d, transform movement-before-hold to d′=D²/(D+h), h′=D+h−d′. This shifts part of the movement into the hold while remaining invertible. Inverse uses S=d′+h′, D=√(d′S), h=S−D. This is a synthetic stressor, not a claim that XCTest does this.
- rand: independent per-clip onset ε∼U(−25,+25)ms; no deterministic warp.
- detnoise: deterministic warp plus independent Gaussian onset SD5ms and independent3% duration multiplicative Gaussian noise.
- collapse: force first movement to50ms and add the removed duration to the hold. Only d+h survives, so original move/hold split is non-identifiable.

Render72 frames at60Hz,16×16 pixels, Gaussian blob σ=.035 normalized, six-frame declining trail, iid pixel noise SD.005. Input features are framewise fourth-power intensity centroids, intensity mass and maximum, computed from pixels without ground-truth coordinates/times. An MLP288→192→96→4 with SiLU uses whole-clip context; labels are the original commanded timing. Standardize training features/labels, AdamW lr.001/weight decay.001, batch128,70 epochs. No held-out tuning or early stopping. Each condition has2400 training and600 held-out examples for each seed0,1,2. Command sampling, executor disturbances, pixels and replay have separate deterministic RNG streams. Same prior/geometry across conditions, but pixel/executor draws are not paired training examples.

All15 fits completed in54.26s. Seed means and sample SDs are descriptive, not confidence intervals for device deployment. Seeds0/1/2 and600 held-out clips each are modest evidence of reproducibility within this toy.

## Metrics and semantics

Predict command ĉ; evaluate command trajectory against c, nominal replay E_nom(ĉ) against observed E(c), and fresh replay E(ĉ;independent ε′) against observed E(c;ε). Fresh replay matters for stochastic executors. Also replay the exact true command with fresh noise, isolating repeatability from model errors.

Use240Hz evaluation timestamps over the1.2s clip. Report mean Euclidean XY error on overlapping contact intervals. To prevent a misleading low XY mean hiding contact errors, pass additionally requires≥95% of **union-contact** samples with both active and distance≤.03, and onset/release each within2/60s. The pass metric is a clearly defined toy analogue, not the production decoder/evaluator.

The same observed trajectories can be interpreted as human raw touches h=E(c) rendered by an identity human input path. On those in-support videos the desired same-executor label is c, not h. `human_raw_error` compares predicted command directly with h; nominal/fresh replay compare executed command with h. This deliberately removes appearance and coverage shift so that action-semantic bias cannot be blamed on domain shift. No claim is made about actual human-video generalization.

## Analytical checks and limits

`verify.py` checks deterministic inverse to machine precision and constructs identical rendered collision pairs: cA=(.15,.14,.18,.20), cB=(.15,.26,.06,.20) under collapse. Both produce exactly the same pixels (shared renderer-noise seed) but move/hold durations differ120ms. Any estimator has balanced-pair per-duration RMSE≥60ms; no model/data quantity recovers which original command occurred. A canonical command can still replay the same effect. For nonlinear equivalence classes the squared-error mean need not be a valid inverse: E(c)=c² with equally probable c=±1 gives prediction0 and incorrect replay. Therefore command ambiguity is fatal to unique recovery, but not necessarily to effect imitation; right-inverse feasibility and loss/decoder choice matter.

Known executed timing also yields an oracle for rand: conditional onset mean is .5[max(.10,z−.025)+min(.24,z+.025)]. Across20000 new samples this gives11.07ms MAE/13.14ms RMSE versus learned14.47ms onset MAE. Thus learned errors are partly information loss and partly model fit, not a measured universal noise floor. The oracle optimizes parameter MSE, not trajectory pass.

Another check constructs two whole clips with identical t−2..t+1 pixel windows but different inverse command positions at200ms, separated.06348 normalized. A strictly local head cannot fit both within.03 even though the whole clips are distinguishable and the executor invertible. **This does not prove the current CurvedModel is strictly local**: its GroupNorm receives five-dimensional B,C,T,H,W activations and transmits clip-wide means/variances. Temporal convolution support is short, but global normalization statistics leak wider information. A follow-on should compare explicit long-context conditioning to this implicit path and test transfer when clip length/background changes.

Rendering is simple, path geometry known, trails distinguishable, no board occlusion/scenery/codec artifacts, no iOS/park/device changes, no Model2 downstream evaluation. These favorable conditions demonstrate possibility and failure mechanisms, not a qualification pass or recommended noise allowance.

## Qualification implication

Separate actual executor E from renderer R and game state s: video=R(E(c),s). A repeatable command→video mapping is insufficient if different physical touches collapse under R (occlusion or quantization), or if the learned model sees human inputs outside the image/support of the training mapping. Same-executor imitation needs a feasible right inverse reproducing the relevant touch/game effects; original-command or executor-independent human-touch recovery additionally needs identifiability. Condition E/R on device/build/park when they differ; do not assume stationarity.

Keep the0.03 contact/timing outcome target as an evaluation gate, compare observed traces with a held-out predicted forward transfer model, measure fresh repeatability, quantify measurement uncertainty, test human supported coverage, and state whether labels mean **executor commands** or **raw human contacts**. Do not replace fidelity gates with an unmeasured assertion of determinism. The previously suggested8ms SD/one-frame95% rules are engineering proposals, not literature-derived guarantees, and must be validated against speed-dependent replay/task outcomes.

## Reproduce

Use the repository’s sole interpreter with `-B`, and invoke this directory’s `study.py` (defaults run all15 fits) then `verify.py`. Both write only beside themselves. `summary.json` contains the three-seed descriptive aggregates derived directly from `results.json`; `run.log` preserves fit losses and per-run metrics.
