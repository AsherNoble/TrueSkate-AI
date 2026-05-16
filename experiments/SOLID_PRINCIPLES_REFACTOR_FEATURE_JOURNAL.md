# SOLID Principles Refactor — Feature Journal

## Background
- Codebase reviewed against SOLID; rated "moderate adherence, pragmatic not strict"
- Strong: focused modules (`sim/gestures.py`, `rl/cmaes/action_param.py`), data-driven
  reward/curriculum (OCP via `Curriculum.from_json`)
- Weak: orchestration god functions + collector/worker coupling
- Refactor motivated by upcoming feature work — code must stay workable as features land

## Assessment (verified 2026-05-16)
Read `cmaes_optimizer.py`, `ppo/trainer.py`, `ppo/collector.py`, `device_worker.py`,
`cmaes/curriculum.py`. Every cited deviation confirmed real:

- **SRP** — `cmaes_optimizer.run()` (lines 148–486) and `ppo/trainer.run_training()`
  (177–538) are god functions: config, worker lifecycle, scheduling, logging,
  checkpoints, metrics, and the optimization/training loop all inline.
- **ISP/DIP** — `ppo/collector.py` reaches into `DeviceWorker` privates: imports
  `_ALL_DEAD_TIMEOUT`, mutates `_failure_streak`/`_dead_since`, calls `_try_revive`.
- **DIP** — training loops depend directly on concrete `DeviceWorker`/`DEVICES`.
- **OCP** — reward/curriculum is genuinely clean; leave it alone.

### Sharper finding (under-stated in original assessment)
`_failure_streak` / `_dead_since` have **two owners**:
- CMA path: `DeviceWorker.evaluate()` never raises — tracks failures internally
  (`device_worker.py:436–441`).
- PPO path: `_collect_one()` raises — collector tracks failures externally
  (`collector.py:214–220`).
Same private state, maintained in two places by two owners depending on entry point.
This is a latent bug, not just an interface smell — it will bite any feature that
touches failure/revive logic.

## Urgency Assessment
- **Not risky to leave** — pipeline is operational; this is maintainability debt,
  not a correctness bug (except the dual-ownership item below).
- **Collector/worker fix: do soon** — the `_failure_streak` dual-ownership is the one
  item with real bug potential. Cheap, contained, independently testable.
- **God-function extraction: nice-to-have** — friction, not risk. Priority depends on
  which planned features touch the training loop (logging/metrics/checkpointing). If
  they do, extract those services first so features land once, not twice (cmaes + ppo).

## Plan
1. Collector/worker lifecycle fix — add explicit `DeviceWorker` lifecycle methods
   (`record_success` / `record_failure` / `maybe_revive`); route both CMA and PPO
   paths through them. Removes private-state access and the dual-ownership bug.
2. Extract orchestration services from `run()` / `run_training()`:
   - 2a. `RunLogger` — unified run-folder + JSONL sink. **(done)**
   - 2b. `DeviceWorker.timed_reset()` — dedup the duplicated timed-reset helper. **(done)**
   - 2c. `WorkerPool` — encapsulate worker list, connect/disconnect, all-dead abort. **(done)**
   - 2d. Metrics reducer — extract the ~40-line `sum(...)/n` block in `run_training`. **(done)**
3. Leave reward/curriculum as-is.

Principle: keep core CMA-ES / PPO logic intact; refactor only the orchestration shell.

## Planned features driving this work
- OCR refactoring/improvements — lives in the reward/curriculum layer (already clean);
  largely orthogonal to the orchestration refactor.
- Remote training monitoring — hooks directly into logging/metrics. Makes the
  `RunLogger` extraction (step 2) high-value: a single sink both cmaes/ppo write
  through, which remote monitoring can tee to a network endpoint.

## Progress Log
- 2026-05-16 — Assessment verified against source. Journal created.
- 2026-05-16 — **Step 1 complete (collector/worker lifecycle fix).**
  Added `DeviceWorker.record_success()` / `record_failure()` / `maybe_revive()`
  (renamed from `_try_revive`) and read-only `dead_since` property; made
  `ALL_DEAD_TIMEOUT` a public module constant. Routed both `evaluate()` (CMA path)
  and `ppo/collector.py` (PPO path) through these. `_failure_streak` / `_dead_since`
  now have a single owner — the dual-ownership bug is resolved. Behavior preserved
  exactly; the only intentional change is `record_success()` now also clears
  `_dead_since` on the CMA path (previously only the streak), which is harmless
  (`_dead_since` is unread on that path) and makes the two paths consistent.
  Verified: syntax/compile, clean import, lifecycle behavior unit-checked.
- 2026-05-16 — **Step 2a/2b complete (RunLogger + timed_reset dedup).**
  New `rl/run_logger.py`: `RunLogger` owns the run folder + line-buffered JSONL
  log; `write()` is the single sink (the hook point for remote monitoring).
  Removed duplicated `_open_log`/`_open_run_log` and `_write_log`/`_write_jsonl`
  from both orchestrators; both now use `RunLogger`. Added
  `DeviceWorker.timed_reset()` and removed the twice-duplicated
  `_timed_worker_reset`/`_timed_reset` helpers. Dropped now-unused `import json`
  from `trainer.py`. Behavior preserved (same folder layout, same JSONL output).
  Verified: compile, no leftover refs, `RunLogger` round-trip + all orchestrator
  modules import clean. Steps 2c (`WorkerPool`) and 2d (metrics reducer) pending.
- 2026-05-16 — **Step 2d complete (metrics reducer).**
  New `rl/ppo/metrics.py`: `RolloutSummary` dataclass + `summarize_rollouts()`.
  Replaced the ~60-line inline rate/mean/device-summary block in `run_training`
  with one `summarize_rollouts(...)` call; the `update_summary` record and the
  per-update print line now read from `summary.*`. `mean_reward`/`max_reward`
  computed from `RolloutResult.reward` (identical to the old tensor path).
  Behavior preserved.
- 2026-05-16 — **Step 2c complete (WorkerPool).**
  New `rl/worker_pool.py`: `WorkerPool` owns the DeviceWorker fleet — construction,
  `connect_all`/`disconnect_all`, `revive_dead`, `raise_if_all_dead`; iterable +
  indexable so dispatch loops are unchanged. Both orchestrators now build a
  `WorkerPool` instead of a raw list and delegate connect/disconnect to it.
  `collect_rollouts` takes `pool: WorkerPool` and delegates the revive + all-dead
  abort (previously inlined). `AllWorkersDeadError` subclasses `RuntimeError`, so
  existing `except RuntimeError` callers are unaffected. `DeviceWorker`/`DEVICES`
  remain exported from `device_worker.py` for the standalone inspect scripts.
  Verified: compile, all modules import clean, `collect_rollouts` signature now
  takes `pool`, `WorkerPool` + `summarize_rollouts` behavior unit-checked.

## Outcome
Steps 1 and 2 complete. The orchestration shell is now factored into reusable
services (`RunLogger`, `WorkerPool`, `summarize_rollouts`) shared by the CMA-ES
and PPO loops; `run()` / `run_training()` coordinate strategy rather than owning
logging, worker lifecycle, and metric reduction. Core CMA-ES / PPO math untouched.
Not yet exercised against live devices — needs a smoke run before the next
training session. New shared modules have no automated tests; behavior was
verified by ad-hoc checks. Worth adding pytest coverage for `summarize_rollouts`
and `WorkerPool` alongside the OCR test work.
