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

## ARCH-006 — Retired autonomous collection fixer

- Description: Laptop watchdog that launched a Claude recovery agent when collection appeared stale, its agent prompt, and its launchd template. Retired at the operator's request; stopped collection is intentional, not an incident to repair automatically.
- Commit: `8ff3720a1bfc9873b501459ad97c26dce243cf22`
- Tag: `archive/autofixer-20260909`
- Paths: [scripts/ops/xr_watchdog_spawn_claude.sh](https://github.com/AsherNoble/TrueSkate-AI/blob/8ff3720a1bfc9873b501459ad97c26dce243cf22/scripts/ops/xr_watchdog_spawn_claude.sh), `scripts/ops/xr_fix_agent_prompt.md`, `scripts/ops/com.trueskate.xrwatchdog.plist` in the same tree.
- Recovery: `git worktree add --detach /absolute/new/arch-006 8ff3720a1bfc9873b501459ad97c26dce243cf22`; do not reinstall or enable the fixer without new explicit authorization.
- Artifacts: Existing logs remain at `/Users/ashernoble/.claude/xr_watchdog.log`; the exact installed plist was moved to the laptop repository's ignored `tmp/com.trueskate.xrwatchdog.retired-20260909.plist`. Neither is a corpus backup.

## ARCH-007 — Research journal entries before 2026-09-12

- Description: The complete research journal (32 dated entries) before it was trimmed to its 30-entry cap. The removed entries are 2026-09-04 (Model 1 evaluation and scaling protocol), 2026-09-08 (BC reconciliation and repository reorganisation) and 2026-09-09 (unintended collection autostart retired).
- Commit: `2cf396f19d7f35560bac8768bb09b0c9efe3a2f5`
- Tag: `archive/journal-20261001`
- Paths: [research/JOURNAL.md](https://github.com/AsherNoble/TrueSkate-AI/blob/2cf396f19d7f35560bac8768bb09b0c9efe3a2f5/research/JOURNAL.md)
- Recovery: `git show 2cf396f19d7f35560bac8768bb09b0c9efe3a2f5esearch/JOURNAL.md`
- Artifacts: None beyond the journal text; the entries link to experiment records that remain in the active tree.


## ARCH-008 — Complete journal before the linear duration audit entry

- Description: The complete 30-entry journal before adding the 2026-10-03 blinded linear duration × length audit. The removed 2026-09-12 human onset timing audit entry remains preserved here and in its experiment record.
- Commit: `ac74a7fe887bb98a791af6fb2d95b488f31f44f9`
- Tag: `archive/journal-before-linear-audit-20261003` (remote tag verified before trimming).
- Paths: [research/JOURNAL.md](https://github.com/AsherNoble/TrueSkate-AI/blob/ac74a7fe887bb98a791af6fb2d95b488f31f44f9/research/JOURNAL.md)
- Recovery: `git show ac74a7fe887bb98a791af6fb2d95b488f31f44f9:research/JOURNAL.md`
- Artifacts: The tag preserves journal text, not ignored recordings. The removed entry's evidence remains in `research/experiments/M1-TIMING-20260912.md` and `research/evidence/M1-TIMING-20260912/`.

## ARCH-009 — Correction to ARCH-007 recovery command

- Description: ARCH-007 remains byte-for-byte unchanged. Its recovery command omitted the colon and the first character of research; use the corrected command below. The curved audit original ARCH-009 independently records the same correction.
- Commit: `2cf396f19d7f35560bac8768bb09b0c9efe3a2f5`
- Tag: `archive/journal-20261001`
- Paths: [journal](https://github.com/AsherNoble/TrueSkate-AI/blob/2cf396f19d7f35560bac8768bb09b0c9efe3a2f5/research/JOURNAL.md)
- Recovery: `git show 2cf396f19d7f35560bac8768bb09b0c9efe3a2f5:research/JOURNAL.md`
- Artifacts: Journal text only.

## ARCH-010 — Integration index for docs/linear-length-audit-journal

- Description: Original feature tip preserved and GitHub-verified before consolidation; original worktree: /private/tmp/trueskate-audit-journal. Original archive IDs remain unchanged under this tag.
- Commit: `781afa22beaf3007a4302d8d951b8375ab80f301`
- Tag: `archive/worktree-20261005/docs/linear-length-audit-journal`
- Paths: [original archive](https://github.com/AsherNoble/TrueSkate-AI/blob/781afa22beaf3007a4302d8d951b8375ab80f301/research/ARCHIVE.md), [source tree](https://github.com/AsherNoble/TrueSkate-AI/tree/781afa22beaf3007a4302d8d951b8375ab80f301/research/) 
- Recovery: `git worktree add --detach /absolute/new/arch-10 781afa22beaf3007a4302d8d951b8375ab80f301`
- Artifacts: 0 ignored regular files (0 bytes) verified under `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/trueskate-audit-journal`. Full original paths, sizes, SHA-256 and symlink targets: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/manifest.json`. Caches and environments excluded; root `.venv` retained. External/rig artifacts retain original locations in experiment records and are not backed up by Git.

## ARCH-011 — Integration index for research/curved-execution-audit

- Description: Original feature tip preserved and GitHub-verified before consolidation; original worktree: /private/tmp/trueskate-curved-audit. Original ARCH-008 means the conjoined-gesture journal, and ARCH-009 is its ARCH-007 recovery correction; these are qualified by this tag and are distinct from main IDs.
- Commit: `ecac1ddccf328f8efcdf9cce8147693ac0f67870`
- Tag: `archive/worktree-20261005/research/curved-execution-audit`
- Paths: [original archive](https://github.com/AsherNoble/TrueSkate-AI/blob/ecac1ddccf328f8efcdf9cce8147693ac0f67870/research/ARCHIVE.md), [source tree](https://github.com/AsherNoble/TrueSkate-AI/tree/ecac1ddccf328f8efcdf9cce8147693ac0f67870/research/) 
- Recovery: `git worktree add --detach /absolute/new/arch-11 ecac1ddccf328f8efcdf9cce8147693ac0f67870`
- Artifacts: 9 ignored regular files (12717392 bytes) verified under `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/trueskate-curved-audit`. Full original paths, sizes, SHA-256 and symlink targets: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/manifest.json`. Caches and environments excluded; root `.venv` retained. External/rig artifacts retain original locations in experiment records and are not backed up by Git.

## ARCH-012 — Integration index for research/curved-gestures

- Description: Original feature tip preserved and GitHub-verified before consolidation; original worktree: /Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/curved-gestures-worktree. Original ARCH-008 means the pre-curve journal, distinct from main ARCH-008.
- Commit: `3f6709ebeeee42d3acbae0a732b87eec1580382e`
- Tag: `archive/worktree-20261005/research/curved-gestures`
- Paths: [original archive](https://github.com/AsherNoble/TrueSkate-AI/blob/3f6709ebeeee42d3acbae0a732b87eec1580382e/research/ARCHIVE.md), [source tree](https://github.com/AsherNoble/TrueSkate-AI/tree/3f6709ebeeee42d3acbae0a732b87eec1580382e/research/) 
- Recovery: `git worktree add --detach /absolute/new/arch-12 3f6709ebeeee42d3acbae0a732b87eec1580382e`
- Artifacts: 12657 ignored regular files (4438079324 bytes) verified under `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/curved-gestures-worktree`. Full original paths, sizes, SHA-256 and symlink targets: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/manifest.json`. Caches and environments excluded; root `.venv` retained. External/rig artifacts retain original locations in experiment records and are not backed up by Git.

## ARCH-013 — Integration index for research/hid-pointer

- Description: Original feature tip preserved and GitHub-verified before consolidation; original worktree: /Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/hid-pointer-worktree. Original archive IDs remain unchanged under this tag.
- Commit: `dd27a22cef9135f3458f58a5fc841c2e1dd48e4e`
- Tag: `archive/worktree-20261005/research/hid-pointer`
- Paths: [original archive](https://github.com/AsherNoble/TrueSkate-AI/blob/dd27a22cef9135f3458f58a5fc841c2e1dd48e4e/research/ARCHIVE.md), [source tree](https://github.com/AsherNoble/TrueSkate-AI/tree/dd27a22cef9135f3458f58a5fc841c2e1dd48e4e/research/) 
- Recovery: `git worktree add --detach /absolute/new/arch-13 dd27a22cef9135f3458f58a5fc841c2e1dd48e4e`
- Artifacts: 0 ignored regular files (0 bytes) verified under `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/hid-pointer-worktree`. Full original paths, sizes, SHA-256 and symlink targets: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/manifest.json`. Caches and environments excluded; root `.venv` retained. External/rig artifacts retain original locations in experiment records and are not backed up by Git.

## ARCH-014 — Integration index for research/hid-review-20261004

- Description: Original feature tip preserved and GitHub-verified before consolidation; original worktree: /Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/hid-review-worktree. Original archive IDs remain unchanged under this tag.
- Commit: `3ccf8bd028320fe60aa30d1f924520734eccb816`
- Tag: `archive/worktree-20261005/research/hid-review-20261004`
- Paths: [original archive](https://github.com/AsherNoble/TrueSkate-AI/blob/3ccf8bd028320fe60aa30d1f924520734eccb816/research/ARCHIVE.md), [source tree](https://github.com/AsherNoble/TrueSkate-AI/tree/3ccf8bd028320fe60aa30d1f924520734eccb816/research/) 
- Recovery: `git worktree add --detach /absolute/new/arch-14 3ccf8bd028320fe60aa30d1f924520734eccb816`
- Artifacts: 0 ignored regular files (0 bytes) verified under `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/hid-review-worktree`. Full original paths, sizes, SHA-256 and symlink targets: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/manifest.json`. Caches and environments excluded; root `.venv` retained. External/rig artifacts retain original locations in experiment records and are not backed up by Git.

## ARCH-015 — Integration index for research/linear-drag-speed-sweep

- Description: Original feature tip preserved and GitHub-verified before consolidation; original worktree: No active worktree. Original ARCH-008 means the pre-curve journal, distinct from main ARCH-008.
- Commit: `c5720aefcc7ad21e7a1f23fb42637781f18fca4c`
- Tag: `archive/worktree-20261005/research/linear-drag-speed-sweep`
- Paths: [original archive](https://github.com/AsherNoble/TrueSkate-AI/blob/c5720aefcc7ad21e7a1f23fb42637781f18fca4c/research/ARCHIVE.md), [source tree](https://github.com/AsherNoble/TrueSkate-AI/tree/c5720aefcc7ad21e7a1f23fb42637781f18fca4c/research/) 
- Recovery: `git worktree add --detach /absolute/new/arch-15 c5720aefcc7ad21e7a1f23fb42637781f18fca4c`
- Artifacts: 0 ignored regular files (0 bytes) verified under `Inherited artifacts retained with linear-speed-worktree`. Full original paths, sizes, SHA-256 and symlink targets: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/manifest.json`. Caches and environments excluded; root `.venv` retained. External/rig artifacts retain original locations in experiment records and are not backed up by Git.

## ARCH-016 — Integration index for research/linear-length-blind-sweep

- Description: Original feature tip preserved and GitHub-verified before consolidation; original worktree: /Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/linear-speed-worktree. Original ARCH-008 means the pre-curve journal, distinct from main ARCH-008.
- Commit: `f90a75fe63f0f13f9eb3ea868dcc808a8c11835d`
- Tag: `archive/worktree-20261005/research/linear-length-blind-sweep`
- Paths: [original archive](https://github.com/AsherNoble/TrueSkate-AI/blob/f90a75fe63f0f13f9eb3ea868dcc808a8c11835d/research/ARCHIVE.md), [source tree](https://github.com/AsherNoble/TrueSkate-AI/tree/f90a75fe63f0f13f9eb3ea868dcc808a8c11835d/research/) 
- Recovery: `git worktree add --detach /absolute/new/arch-16 f90a75fe63f0f13f9eb3ea868dcc808a8c11835d`
- Artifacts: 51 ignored regular files (254236065 bytes) verified under `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/linear-speed-worktree`. Full original paths, sizes, SHA-256 and symlink targets: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/manifest.json`. Caches and environments excluded; root `.venv` retained. External/rig artifacts retain original locations in experiment records and are not backed up by Git.

## ARCH-017 — Integration index for research/xr-jailbreak-20261004

- Description: Original feature tip preserved and GitHub-verified before consolidation; original worktree: /Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/xr-jailbreak-worktree. Original archive IDs remain unchanged under this tag.
- Commit: `3ccf8bd028320fe60aa30d1f924520734eccb816`
- Tag: `archive/worktree-20261005/research/xr-jailbreak-20261004`
- Paths: [original archive](https://github.com/AsherNoble/TrueSkate-AI/blob/3ccf8bd028320fe60aa30d1f924520734eccb816/research/ARCHIVE.md), [source tree](https://github.com/AsherNoble/TrueSkate-AI/tree/3ccf8bd028320fe60aa30d1f924520734eccb816/research/) 
- Recovery: `git worktree add --detach /absolute/new/arch-17 3ccf8bd028320fe60aa30d1f924520734eccb816`
- Artifacts: 0 ignored regular files (0 bytes) verified under `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/xr-jailbreak-worktree`. Full original paths, sizes, SHA-256 and symlink targets: `/Users/ashernoble/Projects/Robotics & hardware/TrueSkate-AI/tmp/worktree-retention-20261005/manifest.json`. Caches and environments excluded; root `.venv` retained. External/rig artifacts retain original locations in experiment records and are not backed up by Git.
