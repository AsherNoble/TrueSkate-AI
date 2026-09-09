# Rig rollout — 2026-09-08

## Source and state

Code rollout baseline: PR #17 merged and was deployed as
`9ff97c5fb29e799ac7cf027fde100cbce9b64600` on 2026-09-09 (Sydney date).
At that verification the stable path pointed to `trueskate-ai-releases/9ff97c5`; dashboard loaded and
disk revisions matched, source was clean and no restart was pending. Both
collectors/watchdogs remained disabled and XR2 WDA was unchanged and ready.
Later documentation-only merges do not imply an unrecorded runtime deployment.

Initial rollout source: `c8c384ffff6cc1bfdf75b3ec43c226886f234b85` (merged BC cleanup).
The paths and process IDs below describe that initial cutover. Later clean
releases may replace the stable symlink; inspect it and the live dashboard's
`/deployment.json` before acting on these historical recovery instructions.

Source cutover is complete. Both collectors are now intentionally disabled by
the operator; recovery checks do not authorize resuming ongoing collection.
See [the subsequent autostart investigation](COLLECTION_AUTOSTART_20260909.md).
XR1 also needs Xcode signing/account attention before a bounded recording test.

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

At the initial cutover, the release linked `.env`, `.venv`, `data`, `logs`, `tmp` and the historical
notebook output directories to the preserved directory. No environment upgrade,
corpus copy, checkpoint conversion or credential publication was performed.
Deployment-local Git exclusions cover only those external runtime links;
tracked source remains fully visible to Git and the dashboard's dirty check.

The later [storage separation](RIG_STORAGE_20260909.md) moved active release
links to a dedicated runtime directory while retaining these original copies.
**Do not delete the preserved directory:** the untouched long-lived service
monitor still depends on the old process environment/cwd and may write old logs;
the historical linked worktree also remains there. Neither same-disk copy is an
independent corpus backup. Do not run `git pull` or agent edits inside a live
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

## Historical recorder investigation and collector configuration

At this stage WDA remained responsive. A read-only official attachment listing found 547
on-device XCTest attachments on XR1 (12 on XR2); nothing was deleted. A running
tunnel daemon alone does not prove its per-device transport remains healthy:
its log also records terminated SSL forwarders. Attachment cleanup/recovery
requires separate approval because recordings never downloaded could be lost.
The live registry check confirmed `activeTunnels: 0` and an empty tunnel map
despite the daemon being alive. Restore per-device tunnel registration before
expecting automatic attachment cleanup to work; do not restart healthy WDA.

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

### Authorized attachment cleanup follow-up

After explicit approval, the operator restarted the root tunnel daemon;
the registry then showed both device tunnels active. The official XR1-only
cleanup deleted all 547 listed XCTest attachments. Saved server corpus data
and XR2's attachments were not deleted.

A single five-second recording probe failed at start: XCTest reported
`Already recording, there can only be one screen recording at a time`.
WDA's `/wda/video` returned null; a direct `/wda/video/stop` also returned null.
Inspection of the installed WDA implementation confirmed stop is a no-op when
its recording promise/ID is missing. Thus clearing attachments did not clear
the orphaned active-recording state. No further start retries or WDA restart
were performed during that probe. The operator subsequently rebooted XR1;
WDA could not rebuild because of an Xcode account/provisioning error.
That hardware recovery is deferred; collection remains off by operator intent.

Fresh source and installed-service snapshots were saved before cutover:

- `/Users/training-server/trueskate-ai-preserved-20260908/tmp/cutover-source-20260908.tgz`
- `/Users/training-server/trueskate-ai-preserved-20260908/tmp/cutover-tracked-20260908.patch`
- `/Users/training-server/trueskate-ai-preserved-20260908/tmp/cutover-services-20260908.tgz`

For source rollback, inspect the current stable symlink and choose a preserved
clean release explicitly. Verify its runtime links before atomically replacing
only the stable symlink, then reload only affected application services.
Do not rename the dirty checkout over the stable path or run the old cutover
script again. Leave WDA untouched and keep collector/watchdog jobs disabled;
recovery alone is never permission to resume collection. Never recursively
delete a release or reset the preserved checkout as part of rollback.

The one-off cutover script at `/Users/training-server/cutover_rig_20260908.py`
contains the executed checks and rollback handling. It is not an idempotent
general-purpose deployment command and must not be rerun after success.
