# Why collection restarted — 2026-09-09

## Evidence

The operator did not intend an ongoing data-collection run; they understood
collection during maintenance to be temporary validation. Three mechanisms
explain why it continued or reappeared:

1. Both installed rig collector plists had `RunAtLoad=true`. Although unloaded
   at inspection, both labels were enabled and could start at a later login/load.
2. The preserved old `scripts/rig_collect.sh` contained an unconditional
   `while true` loop, restarting the collector 30 seconds after any exit. The
   cleaned version already removed that loop, but did not disable login startup.
3. A laptop LaunchAgent, `com.trueskate.xrwatchdog`, was installed and loaded
   with a 900-second interval and `RunAtLoad=true`. Its script launched a fresh
   Claude fixer when either collector appeared stale. Its log explicitly
   records an XR1 relaunch producing session `iPhone_XR_20260907_234013` and
   repeated recovery attempts/escalations. This is direct evidence of an
   autonomous restart path, not just the existence of an unused script.

The maintenance agent also resumed XR1 after validation/cutover by assuming
the pre-existing workload should be restored. That assumption was wrong; the
operator has now explicitly clarified that collection should remain stopped.

## Actions

- Disabled both rig collector labels persistently in `gui/501`; neither is loaded.
- Disabled and unloaded both collection-only fleet watchdogs, so intentional
  inactivity does not generate recovery alerts.
- Disabled and unloaded the laptop fixer in `gui/503`; no active fixer child
  was found. Moved its installed plist out of LaunchAgents into the repository's
  ignored `tmp/com.trueskate.xrwatchdog.retired-20260909.plist`.
- Removed the fixer script, prompt and launchd template from active source after
  publishing a protected archive tag. [Recovery/provenance: ARCH-006](ARCHIVE.md).
- Kept the dashboard, WDA services, recording tunnel and existing storage/offload
  schedules separate. No open-ended collector was started during this investigation.

No user crontab was installed on the rig. The inspected rig user/global launch
directories and shell startup files contained no additional collection starter.
Manual collection/supervisor scripts exist, but no corresponding active supervisor
or scheduled handoff was found. This is an inventory of the inspected mechanisms,
not a claim that an arbitrary future agent or manually run script cannot start one.

## Remaining hardware check

The operator rebooted XR1 after the authorized deletion of its 547 temporary
attachments. Its WDA did not become usable, so no new recording was started.
An existing Xcode build log reports `No Accounts: Add a new account in Accounts
settings` and no iOS development profile for the WDA runner identifier. This
requires operator signing/account work; exit code 65 alone was not the diagnosis.
XR2 WDA remained healthy and was not restarted. Bounded XR1 recording verification
is deferred until its WDA is available. Collection remains off afterward as well.

## Future operating rule

An idle or stale collector is not authorization to collect. Any future ongoing
run needs an explicit workload/duration decision. Bounded maintenance probes
use isolated output paths and must terminate themselves; they do not enable
launchd collection jobs or autonomous recovery agents.
