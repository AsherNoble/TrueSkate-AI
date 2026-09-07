# Repository instructions

## Current direction

Behavioural cloning is authoritative. Read `research/STATUS.md` and the short
`research/JOURNAL.md` for relevant research work. Model 1 linear/scaling is
current; hold, heatmap and temporal variants remain runnable experiments.
Model 2 is unfinished. PPO/CMA-ES implementations are retired in tagged history.

## Navigation and conventions

- `README.md`: architecture; `docs/WORKFLOWS.md`: runnable commands.
- `GESTURES.md`: authoritative gesture, bounds, timing and schema contracts.
- `DEPLOYMENT.md`: rig operation, compatibility launchers and rollback.
- `.venv` is the only virtual environment; run `python -m pytest` from repo root.
- Temporary output goes in `tmp/`; corpus/checkpoint artifacts stay out of Git.
- Use absolute filesystem paths in tools. Do not rely on tilde expansion.
- Commit messages: `type: message`, one coherent commit at a time.
- Prefer terse factual updates: what changed, evidence, next step.
- Preserve dirty work. Make agent changes in separate worktrees/branches;
  treat the rig checkout as deployment state, not an untracked development line.

## Rig invariants

- XRs normally attach to `training-server`; use
  `tailscale ssh training-server@training-server`. Do not assume local USB.
- UDIDs come from `.env`. XR1 WDA/Appium: 8100/4723; XR2: 8103/4726.
- Keep XCTest segments at one minute: stop returns the full recording as base64.
- Root `com.trueskate.remotexpc-tunnel` must be running. Otherwise recording
  attachments accumulate and wedge the recorder. Recovery uses the documented
  attachment cleanup tool after confirming the daemon; never hammer recorder start.
- Do not restart a healthy WDA service to test a hypothesis. A running WDA
  may survive an expired signing profile that prevents rebuilding it.
- Respect recorder start-failure caps, gameplay/foreground guards, per-segment
  calibration, and decoded frame-count checks. Never weaken admission to pass a smoke test.
- Training loaders exclude contamination markers. `.menu` is always excluded;
  stricter loaders also enforce their existing calibration and trace contracts.
- Preserve normalised coordinates and curved/overlapping/spin gestures. iPhone
  11 Display Zoom yields 375×812; XR devices expect 414×896. No y offset.
- Notifications belong to the fleet incident/recovery mechanism. No periodic
  failure reminders or agent-generated messages without explicit authorization.
- Collection smoke output must use an isolated directory and honest park provenance.
- No paid cloud training or new research tranche without explicit authorization.

## Research hygiene

- `research/STATUS.md`: current evidence, supported variants, open questions.
- `research/JOURNAL.md`: at most 30 short chronological entries. Tag and index
  the complete previous journal before trimming. Link substantial experiments
  to individual records; preserve existing experiment IDs.
- Durable findings update reference docs or individual experiment records.
  Historical events do not become current operating instructions automatically.
- `research/ARCHIVE.md` is byte-for-byte append-only. Append corrections that
  reference an existing ID; never edit/reorder/delete published entries.
- Every retirement records full SHA, tag, original paths, recovery command and
  external artifact locations. Verify GitHub preservation before deleting source.
- Run `python scripts/ops/check_archive.py --base origin/main` before merging.
  Changing the checker/workflow is a deliberate policy change requiring review.
- Git tags do not back up ignored datasets, local logs, `.env`, or cloud volumes.
- Research holdouts are not refactor tests. Use synthetic fixtures and existing
  offline tests; preserve model/checkpoint and dataset schema compatibility.
