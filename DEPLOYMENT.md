# Rig deployment and rollback

The XR fleet is on `training-server`, normally checked out at
`/Users/training-server/trueskate-ai`. Reach it via
`tailscale ssh training-server@training-server`.

The stable path now points to the clean `c8c384f` release. The original dirty
checkout is preserved separately and backs the runtime data/environment links.
See the [2026-09-08 rollout record](research/RIG_ROLLOUT_20260908.md) for exact
paths, loaded-service exceptions and rollback. Do not delete the preserved
directory or edit/pull into the active release.

## Services

| Service | Purpose |
|---|---|
| `com.trueskate.services` | Appium/WDA/iproxy health monitoring |
| `com.trueskate.dashboard` | Corpus preview and progress on port 8400 |
| `com.trueskate.collect.xr1`, `.xr2` | Experimental mixed SLS collection, when enabled |
| `com.trueskate.watchdog.xr1`, `.xr2` | Fleet incident/recovery monitoring |
| `com.trueskate.autooffload`, `.storageguard` | Configured storage/offload operations |
| root `com.trueskate.remotexpc-tunnel` | Recording attachment cleanup prerequisite |

XR1 uses WDA 8100/Appium 4723; XR2 uses 8103/4726. Confirm USB and WDA health
before collecting. The tunnel daemon must be reachable before recording.
Keep segments at one minute. Service templates live in `scripts/ops`;
the pre-migration installed definitions are preserved in ARCH-004.

## Deploy a committed revision

1. Check the current SHA, tracked/untracked source changes, active jobs, USB,
   WDA endpoints and tunnel. Preserve any new source before deployment.
2. Stage the candidate in a separate worktree. Reuse the existing environment
   only when dependencies match; do not upgrade a live environment during a run.
3. Run offline checks and one bounded collector segment per XR into `tmp/`.
   Use the actual park name, or explicitly label an unclassified validation
   scene. Verify calibration, decoded frame counts and strict loader admission.
4. Switch operational callers only between collection segments. Preserve
   installed service definitions and the previous committed revision. Do not
   restart healthy WDA services just because Python files moved.
5. Record deployed SHA, source-tree status and which long-lived services have
   actually loaded that revision. Verify dashboard and collector output.

Old entrypoint paths remain compatibility launchers. Canonical paths live under
`scripts/collection`, `scripts/model1` and `scripts/model2`. Most legacy shell
wrappers still expect the stable rig root; do not launch them from a staging
worktree and assume they select that worktree. Invoke the Python collector
directly for staging checks.

### Idle navigation versus blocking menus

The neutral five-button bottom navigation can appear over usable idle gameplay
(operator confirmation, 2026-09-08). It is not sufficient evidence that gestures
are blocked. For an operator-confirmed scene, the collector's explicit
`--allow-idle-navigation` option ignores only that signature and records the
choice in the segment manifest. Replay/camera, editor and foreground checks
remain enabled. Do not substitute `--no-menu-guard`.

This option does not relax calibration or dataset admission and does not
reclassify existing `.menu` samples. Frame filtering remains conservative;
changing historical corpus labels requires a separate validated review.

`scripts/rig_collect.sh` now lets the collector's start-failure cap terminate
the process. It does not endlessly restart a wedged recorder or send repeated
notifications; resolve the tunnel/attachment incident before restarting it.

## Recovery

For a code regression, stop only the affected collector/dashboard at a safe
boundary and restore its previous code path and saved service definition.
Do not reset a dirty checkout or erase data. A tag can be inspected safely with
`git worktree add --detach /absolute/new/recovery-path <tag>`.

For recorder failures, confirm the root tunnel, then inspect
`scripts/recover_remotexpc_attachments.sh --help` and use its dry-run before
targeted cleanup. Do not restart healthy WDA: expired signing may prevent it
from rebuilding. Never broaden a source migration into phone reconfiguration.

Keep `.env`, datasets, cloud volumes and checkpoints independent of source
deployment. No corpus migration or cloud retraining is required by this refactor.
