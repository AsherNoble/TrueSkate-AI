# Curved Model 1 operations

Read the [agreed protocol](../research/protocols/model1_curved_collection_v1.md)
for thresholds and stopping rules. The installed park library is permanently
recorded in [config/parks.json](../config/parks.json), with exact UI names.

Use the repository `.venv`. All output belongs in an isolated, persistent run
directory outside Git. The following `CURVED_ROOT` is an example; use one actual
absolute directory for the entire experiment, including retries and production.
Rig execution uses an isolated committed release, not edits to the active rig
checkout. Keep collection supervisors disabled.

## Prepare and collect a qualification round

```bash
export CURVED_ROOT=/Users/training-server/trueskate-ai-runtime/data/curved-model1-v1
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" init
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" start-audit --wda-revision <verified-live-revision>
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" status
```

`init` refuses an existing directory and records the five selected parks. It does
not contact a device. Obtain the exact healthy WDA timing revision through the
existing read-only status/timing endpoint; never rebuild WDA to discover it.
`start-audit` freezes collection code, settings and that WDA revision. Find the
session IDs in the ledger using the following read-only helper:

```bash
python -c 'import os; from pathlib import Path; from trueskate_ai.model1.curved.ledger import Ledger; l=Ledger(Path(os.environ["CURVED_ROOT"])); print(l.sessions()); l.close()'
```

The operator loads the named park and saved waypoint on the specified XR. Keep
the phone awake, with True Skate frontmost. Execute one explicit session:

```bash
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" collect \
  --session <audit-session-id> --park 'Skatepark: Inbound' --waypoint 1 \
  --settings-note '<saved waypoint and fixed game settings>' \
  --device iPhone_XR --wda-revision <verified-live-revision> \
  --env-file /Users/training-server/trueskate-ai/.env --max-minutes 2
```

Two clean minutes can admit twenty clips. Rejected candidates still spend the
attempt budget; continue the same session with another explicit bound if needed.
A recorder or calibration failure stops immediately. Resolve it before continuing;
do not restart a healthy WDA. The per-minute JSON output states admissions and
global attempts spent. There are no spoken or push notifications.

The run lock prevents two commands mutating one experiment concurrently.
After a crash, inspect preserved recordings and pending IDs:

```bash
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" reconcile
```

Complete, verified sample evidence can recover an admission. To reject unresolved
outcomes and cancel unexecuted plans without ever replaying the commands, rerun
with `--abandon-unresolved`. This preserves spent attempts. To declare a failed
technical round, use `fail-audit --reason '<diagnosis>'`; after a fix, start the
next round with `start-audit`. Only rounds whose 100 clips received a human
verdict count towards the total five; `status` reports `reviewed_rounds`.
Technical failures still require a revised pipeline before the next round.

## Audit and qualification

After 20 admitted clips in each park:

```bash
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" seal-audit --round 1
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" view-audit --round 1 --port 8765
```

`view-audit` serves the review UI (`model1/curved/review.py` and `review.html`)
on localhost only. `viewer.py`/`viewer.html` stay byte-identical because sealed
rounds fingerprint them and production must reuse that fingerprint; the review
UI is outside the fingerprint and changes neither the export nor scoring.

Prefer reviewing on the laptop. Copy `audit-N/manifest.json` and every listed
sample's `clip.mp4` and `meta.json` into a local directory with the same
relative paths, then run
`python -m trueskate_ai.model1.curved.review --root <copy> --round N --open`.
The server re-verifies every clip and metadata hash against the sealed manifest
before it starts, decodes each clip once (well under a second on an M2) and
prefetches the next clips. A rig-hosted review still works through SSH
forwarding, but it ships every full-resolution frame over the tunnel.

Clips are grouped by park; the first-contact mark (S), judgement (1/2/3),
systemic flag (D) and reason save automatically in the browser under the manifest hash, and
Save & next (Enter) moves to the next incomplete clip. Export is enabled once
every clip satisfies the same completeness rules as import. A local server
has no `review.json` to watch; stop it with Ctrl-C after the rig import.

Export all 100 ratings and import that exact file:

```bash
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" import-review /absolute/path/to/export.json
```

Import verifies the manifest, reviewed file hashes, uniqueness, required marks,
timing threshold and ratings. It writes immutable review evidence and causes the
viewer to exit. Also close any SSH forwarding process. If an export is invalid,
stop the viewer when its path is returned; repair the review before restarting it.
There is no automatic production launch after a pass.

## Production and frozen selection

```bash
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" prepare-production --wda-revision <qualified-revision>
```

This lists fifty predetermined sessions, ten per park, with 280 admitted clips
per session. Use `collect` with the correct park/session and `--max-minutes 28`
or a smaller bound. Continue under the same session ID if fewer than 280 clips
are admitted. Device, waypoint and settings must remain constant inside that
session. The exact qualified pipeline fingerprint must match on every run.

```bash
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" freeze-dataset
```

This requires the complete separate 14,000 pool. It selects 12,500 with the agreed
park and partition counts and writes `dataset.json`. Surplus and originals stay
in place. The 20,000 candidate ceiling is enforced before each dispatch.

## Freeze human clips and train

Create a JSON list of ten human clip records, two per selected park:

```json
[{"park":"Skatepark: Inbound","video":"/absolute/path/human-01.mp4",
  "single_contact":true,"in_scope":true}]
```

Each input must already be a 120-native-frame 60 Hz clip, containing one contact
inside the MVP scope. Freeze the full ten-item list before training:

```bash
python scripts/collection/curved_model1.py --root "$CURVED_ROOT" freeze-human /absolute/path/human-clips.json
python scripts/model1/train_curved_model1.py train \
  --manifest "$CURVED_ROOT/dataset.json" --human "$CURVED_ROOT/human.json" \
  --out /absolute/path/curved-run --device cpu
```

The CPU command is runnable but the intended full run uses Modal. Add `--resume`
to recover the last complete epoch; never change the recipe or inputs in place.
`latest.json` identifies the validation-selected checkpoint. A separate `test`
command with the same manifest and output exposes the test partition once.
If interrupted after exposure, preserve that fact and obtain an explicit research
decision before another exposure. There is no automatic recollection.

After retrieving the model run directory, generate qualitative human review:

```bash
python scripts/model1/train_curved_model1.py human \
  --manifest "$CURVED_ROOT/dataset.json" --human "$CURVED_ROOT/human.json" \
  --out /absolute/path/curved-run --device cpu
open -a "Google Chrome" /absolute/path/curved-run/human-review/index.html
```

The page shows the same native video frame twice, with predictions only on the
right. Predicted commands and safety status are exported for inspection; this
command never dispatches them to a phone.

## Modal: prepare first, approve actual spending separately

```bash
python scripts/model1/curved_modal.py bundle --manifest "$CURVED_ROOT/dataset.json" \
  --human "$CURVED_ROOT/human.json" --out /absolute/path/curved-bundle
python scripts/model1/curved_modal.py quote --storage-gib <bundle-size>
```

These are offline. Only after explicit storage approval:

```bash
python scripts/model1/curved_modal.py upload --bundle /absolute/path/curved-bundle \
  --approved-storage-usd-month <approved-monthly-amount>
```

Only after explicit benchmark budget approval (the amount must cover the stated
runtime estimate plus one dollar of startup/build headroom):

```bash
python scripts/model1/curved_modal.py benchmark --bundle-hash <exact-bundle-hash> \
  --run curved-v1-benchmark --max-seconds 900 --approved-budget-usd <approved-amount>
```

Use its returned JSON with `quote --benchmark /absolute/path/benchmark.json`
to estimate the 40-epoch run, validation and final test. Approve that estimate
before invoking `train` with a new run name, budget and runtime. Training saves
and commits after each complete epoch. Use `--resume` after interruption.
`test` is a separately bounded compute invocation against the completed run.
No `modal run` is needed; the Python wrapper validates the budget first.

The volume paths are `trueskate-curved-model1-v1/<bundle-hash>` and
`trueskate-models/curved-v1/<run>`. Runtime limits and estimates are not provider
billing caps: builds, startup, storage, transfers and account allowances vary.
