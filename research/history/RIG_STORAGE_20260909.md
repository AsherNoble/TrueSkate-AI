# Rig runtime storage and backup audit — 2026-09-09

## Layout

| Absolute path below `/Users/training-server` | Role |
|---|---|
| `trueskate-ai` | Stable symlink to the selected clean source release |
| `trueskate-ai-releases/9ff97c5` | Source release at storage cutover; runtime links point directly to the directory below |
| `trueskate-ai-runtime` | Active `.env`, `.venv`, `data`, `logs`, `tmp`, and notebook output directories |
| `trueskate-ai-preserved-20260908` | Original dirty source and retained original storage, not a deletion candidate |

Subsequent documentation releases may change the source pointer without changing
runtime storage. Inspect the stable symlink and dashboard `/deployment.json` for
the actually loaded revision; do not infer it from this historical release name.

Runtime storage was copied, not moved or deleted. Its root is mode 0700; `.env`
was not printed, published or uploaded. Approximately 32 GiB of corpus data,
1.2 GiB of environment, 282 MiB of logs, 2.2 GiB of temporary output and 56 MiB
of notebook material were present before separation (rounded disk usage).

The historical `tmp/migration-candidate-20260908` Git worktree was deliberately
excluded from the copy: it remains registered in the preserved checkout and
must not be duplicated with shared Git metadata. Its validation outputs and
other temporary records outside that worktree were copied.

## Verification and service boundary

Static runtime files (`data`, `.env`, `.venv`, notebook artifacts/demo/output)
were checked with checksum-based rsync comparison before switching release
links. This is byte comparison, not model evaluation or reuse of a holdout.
Logs and temporary output are snapshots; they are not claimed to remain equal
while long-lived processes continue writing.

Only the dashboard was reloaded. Collection and collection watchdogs stayed
disabled; no recording, training, WDA restart or manual offload was invoked.
The scheduled offloader and storage guard were paused while copying, then their
existing intervals/environment were restored using maintenance plist copies
with `RunAtLoad=false`, avoiding an immediate invocation. Installed login plists
remain unchanged. Loaded copies and the one-off script are retained at:

- `/Users/training-server/trueskate-storage-maintenance-20260909/`
- `/Users/training-server/separate_runtime_20260909.py`

The old service monitor intentionally still has its original code, environment
and cwd loaded and may write logs in the preserved tree. Its normal future
restart will resolve the stable path; do not restart healthy XR2 merely to
remove this exception. The old source, environment, logs and linked worktree
must remain available in the meantime.

## Backup coverage: not established

The rig has no configured Time Machine destination and no external backup disk
was mounted at inspection. Both local copies are on the same internal SSD:
they provide rollback protection, not protection against disk loss.

Read-only Modal inspection confirmed eight project volumes exist, including
`trueskate-corpus`, `trueskate-corpus-v2`, `trueskate-mvp`, `trueskate-models`
and four linear-dataset volumes. Corpus roots and model checkpoint entries
were readable. This establishes availability only, not full byte-for-byte
coverage, retention or independent duplication of cloud data.

A bounded download probe restored the existing
`trueskate-models/basic_linear_model1_recovered_20260903_seed0.pth` to
`/Users/training-server/trueskate-storage-maintenance-20260909/restore-probe-seed0.pth`:
157,645 bytes, observed SHA-256
`78b6f78e85ef177c28acb1eec4cf32645d91f9d7afeb8b0f50c9a9829951569f`.
The checkpoint was not loaded or evaluated. No independently recorded original
checksum was available for comparison: this proves that one object can be
downloaded, not that every artifact has been backed up or integrity-verified.

Neither corpus volume had a root-name match for any of the 16 local SLS session
directories. The root listings contained 52 entries (`trueskate-corpus`) and
63 (`trueskate-corpus-v2`); `trueskate-models` had 136 root entries. This is a
concrete coverage gap under the offloader's naming convention, not proof that
no transformed/subset copy exists elsewhere. Empty or failed local sessions
are included in that directory count. The read-only audit script is retained
at `/Users/training-server/audit_backup_roots_20260909.py`.

The installed offloader targets `trueskate-corpus-v2`, requires a sampler spin
fraction of at least 0.3 and 45-minute quiescence, and deletes local sessions
after its verification. It is therefore selective migration, not a backup of
all local datasets, logs, credentials or checkpoints. Its sample-count check
is not a cryptographic integrity audit. It was not run manually for this task.

Do not delete either storage copy on the strength of Git tags or cloud volume
names. Before deletion, establish a separately stored backup with an inventory,
integrity verification and a restore test, resolve the live-monitor dependency,
and explicitly handle the historical worktree. A backup destination/retention
decision is still needed; this maintenance did not buy storage or upload `.env`.

## Future release and rollback rule

New releases must link directly to `trueskate-ai-runtime`, not to the preserved
checkout. Before using an older release for source rollback, explicitly verify
or update its runtime links to the current runtime store; its old links may
otherwise select stale copied data. Never merge the two data trees or erase
one automatically. Switch only the source symlink and reload affected services;
keep collection off and WDA untouched.
