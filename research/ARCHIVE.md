# Permanent research archive directory

Append only. Never rewrite existing bytes or reorder entries. Append corrections referencing the original ID. Tags are readable names; full commit-pinned links identify exact source.

To restore a whole historical environment without modifying the active checkout: `git fetch origin --tags`, then `git worktree add --detach /absolute/new/recovery-directory <full-commit-sha>`. Old dependency/tool versions may need reconstruction; preserved source is not a promise that external services still run.

## ARCH-001 — Historical research journals and handovers

- Description: Original journals, experiment queue, interim plans and analyses. Current status and protocols have moved into maintained research documents.
- Commit: `946639a737884b74fc1602d1ec5365730e5cae36`
- Tag: `archive/research-pre-cleanup`
- Paths: [experiments](https://github.com/AsherNoble/TrueSkate-AI/tree/946639a737884b74fc1602d1ec5365730e5cae36/experiments/); follow the repository tree at this SHA for related paths.
- Recovery: `git worktree add --detach /absolute/new/arch-001 946639a737884b74fc1602d1ec5365730e5cae36`
- Artifacts: Historical logs and checkpoint locations are recorded in the original journals; ignored/cloud artifacts are not contained in this tag.

## ARCH-002 — Retired PPO and CMA-ES systems

- Description: Optimisers, reward evaluation and orchestration. Related curricula/, configs/overnight/, deploy/ and original training scripts are in the same tree. Shared gestures/device code was extracted into sim; useful recipe outputs remain in trick_libraries/.
- Commit: `946639a737884b74fc1602d1ec5365730e5cae36`
- Tag: `archive/research-pre-cleanup`
- Paths: [src/trueskate_ai/rl](https://github.com/AsherNoble/TrueSkate-AI/tree/946639a737884b74fc1602d1ec5365730e5cae36/src/trueskate_ai/rl/); follow the repository tree at this SHA for related paths.
- Recovery: `git worktree add --detach /absolute/new/arch-002 946639a737884b74fc1602d1ec5365730e5cae36`
- Artifacts: Tracked logs/runs and trick_libraries are in this snapshot. Untracked training outputs remain on their original machines.

## ARCH-003 — Retired DAL capture and exploratory notebook artifacts

- Description: Historical notebook artefacts, demo images and outputs. Dead DAL implementation is src/trueskate_ai/vision/dal_capture.py, with scripts/data/collect_sls_traces.py, scripts/view_device.py and tools/enable_dal.c in this same snapshot.
- Commit: `946639a737884b74fc1602d1ec5365730e5cae36`
- Tag: `archive/research-pre-cleanup`
- Paths: [notebooks](https://github.com/AsherNoble/TrueSkate-AI/tree/946639a737884b74fc1602d1ec5365730e5cae36/notebooks/); follow the repository tree at this SHA for related paths.
- Recovery: `git worktree add --detach /absolute/new/arch-003 946639a737884b74fc1602d1ec5365730e5cae36`
- Artifacts: Ignored notebooks/models checkpoints remain external; retained spin reference images stay in the active tree.

## ARCH-004 — Training-server source and deployed configuration

- Description: Full rig committed history plus copied modified/untracked source, historical backups and installed user launchd files. scripts/rig_collect.sh and all rig source are in this same tree. See RIG_RECONCILIATION.md for dispositions.
- Commit: `567d0929155494a4905e27f741b2972daf983ec7`
- Tag: `archive/rig-working-20260908`
- Paths: [preservation/deployed-launchagents](https://github.com/AsherNoble/TrueSkate-AI/tree/567d0929155494a4905e27f741b2972daf983ec7/preservation/deployed-launchagents/); follow the repository tree at this SHA for related paths.
- Recovery: `git worktree add --detach /absolute/new/arch-004 567d0929155494a4905e27f741b2972daf983ec7`
- Artifacts: Rig data/, logs/, .env and root daemon remain on training-server. .env and runtime artifacts were intentionally excluded. Cloud volume existence/backups not verified.

## ARCH-005 — Main before behavioural cloning promotion

- Description: Previous authoritative development line, preserved before the normal BC merge.
- Commit: `cb491fc6405c5022700274f5852837aa41106b97`
- Tag: `pre-bc-main`
- Paths: [src](https://github.com/AsherNoble/TrueSkate-AI/tree/cb491fc6405c5022700274f5852837aa41106b97/src/); follow the repository tree at this SHA for related paths.
- Recovery: `git worktree add --detach /absolute/new/arch-005 cb491fc6405c5022700274f5852837aa41106b97`
- Artifacts: Only committed artifacts are preserved; machine-local datasets/checkpoints are independent.

