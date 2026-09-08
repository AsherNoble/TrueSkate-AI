# Rig rollout — 2026-09-08

## Source and state

Deployed source: `c8c384ffff6cc1bfdf75b3ec43c226886f234b85` (merged BC cleanup).

Source cutover is complete. XR1 subsequently hit its recorder start-failure
cap and is currently stopped; recorder recovery is separate from source rollout.

| Path | Role |
|---|---|
| `/Users/training-server/trueskate-ai` | Stable service entry path; symlink to the release below |
| `/Users/training-server/trueskate-ai-releases/c8c384f` | Clean deployment checkout |
| `/Users/training-server/trueskate-ai-preserved-20260908` | Original dirty checkout and shared runtime storage |

The ten modified source files and eleven untracked source/backup files matched
the previously published rig snapshot byte-for-byte. The full original directory
was renamed intact, not reset or cleaned. Its branch remains at
`463316d34b81129986a171920369dd6067e91f7b`; all dirty source remains recoverable.
The linked migration worktree was repaired and still resolves to `292838d`.

The release links `.env`, `.venv`, `data`, `logs`, `tmp` and the historical
notebook output directories to the preserved directory. No environment upgrade,
corpus copy, checkpoint conversion or credential publication was performed.
Deployment-local Git exclusions cover only those external runtime links;
tracked source remains fully visible to Git and the dashboard's dirty check.

**Do not delete the preserved directory:** it is also the live backing store for
data, logs and the environment. This is source preservation, not an independent
backup of the growing corpus. Do not run `git pull` or agent edits inside a live
release; prepare another clean release and switch at a safe boundary.

## Service transition

- XR1's restart wrapper was suspended, then its collector received SIGINT.
  It saved its final 25.11 MB/two-gesture segment and exited normally. Existing
  aligners were allowed to finish before the switch.
- Dashboard, both watchdogs and XR1 collection were reloaded from the stable
  path. The initial cutover did not replace installed plist contents.
- XR2 collection was stopped before rollout and remained stopped; no new
  unbounded collection workload was introduced.
- Scheduled storage/offload invocations were briefly disabled, then re-enabled.
  The migration did not manually invoke offload or change its existing schedule.
- WDA/service-monitor processes remained running: monitor PID 453, XR1 WDA
  PID 61159, XR2 WDA PID 60439. Both WDA endpoints were ready after the switch.
  The root remotexpc tunnel was not restarted.

The long-lived service monitor still has its original Python code loaded. This
is deliberate: restarting it would tear down healthy WDA. Its next ordinary
service launch resolves the stable path to the clean release. Do not confuse
this exception with a claim that every live process reloaded new code.

## Verification

- Exact release checkout with the existing rig environment: 256 tests passed,
  three skipped (one unavailable OpenCV MP4 writer; two optional visual fixtures).
- Earlier bounded collection acceptance on both XRs, including strict linear
  admission and exact decoded frame counts: [migration validation](MIGRATION_VALIDATION.md).
- Live `/deployment.json`: loaded and disk SHA both `c8c384f`, source clean,
  no restart pending. Runtime device and Model 1 imports passed after cutover.
- XR1 relaunched the canonical `scripts/collection/collect_sls_xctest.py`,
  retaining its mixed SLS configuration and 0.5 spin fraction. This is the
  existing mixed collection workload, not a new calibrated Model 1 tranche.
- Its initial post-cutover run reported 20 segment indices and 99 gestures;
  segment 19 was lost on retrieval, followed by recording-start failures.
  The capped collector exited instead of retrying indefinitely. A later run
  also reached its start-failure cap without collecting samples.

## Recorder recovery and collector configuration

WDA remains responsive. A read-only official attachment listing found 547
on-device XCTest attachments on XR1 (12 on XR2); nothing was deleted. A running
tunnel daemon alone does not prove its per-device transport remains healthy:
its log also records terminated SSL forwarders. Attachment cleanup/recovery
requires separate approval because recordings never downloaded could be lost.

The installed XR1 plist now invokes the canonical collector directly, adding
`--allow-idle-navigation` (operator-confirmed idle UI) and
`--heartbeat-path /Users/training-server/trueskate-ai/data/sls_xctest/.collector_heartbeat_iPhone_XR.json`.
This corrects the dashboard's previous use of an old stopped heartbeat from
another run. The 0.5 spin mix and stagger setting remain unchanged. The original
plist is preserved at
`/Users/training-server/trueskate-ai-preserved-20260908/tmp/collector-xr1-before-heartbeat.plist`.
XR1 was left unloaded/stopped after installing the configuration; do not
bootstrap it until recorder recovery is verified. XR2 remains stopped.

## Recovery

Fresh source and installed-service snapshots were saved before cutover:

- `/Users/training-server/trueskate-ai-preserved-20260908/tmp/cutover-source-20260908.tgz`
- `/Users/training-server/trueskate-ai-preserved-20260908/tmp/cutover-tracked-20260908.patch`
- `/Users/training-server/trueskate-ai-preserved-20260908/tmp/cutover-services-20260908.tgz`

For rollback, first drain collectors/aligners and suspend scheduled writers.
Stop only collector/dashboard/watchdog jobs. Verify that the stable path is
exactly the release symlink above, remove only that symlink, and rename the
preserved directory back to the stable path. Repair its migration worktree at
the restored location and restore the original XR1 plist from the backup above
(the legacy source does not support the new idle-navigation flag). Restore
dashboard/watchdog jobs and scheduled enablement; resume collection only after
recorder recovery. Leave WDA untouched. Never recursively delete either
directory or reset the preserved checkout.

The one-off cutover script at `/Users/training-server/cutover_rig_20260908.py`
contains the executed checks and rollback handling. It is not an idempotent
general-purpose deployment command and must not be rerun after success.
