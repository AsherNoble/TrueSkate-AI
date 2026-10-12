# Free-Colab Model 1 benchmark harness

Runs one frozen Model 1 training seed on a free Colab GPU, with an allocation
ledger, verified per-epoch checkpoints in one private Drive folder and a
deliberate runtime-deletion recovery test. Experiment record:
[M1-COLAB-20261006](../../research/experiments/M1-COLAB-20261006.md).

| File | Role |
|---|---|
| `benchmark.py` | Harness run inside the pinned `.venv`: staging, preflight, training via the frozen trainer's checkpoint callback, restoration check, report, verified uploads |
| `supervisor.py` | Runs in the Colab kernel: ledger, budget deadline, pinned install, input pull, runtime release |
| `build_notebook.py` | Generates the notebook; `--check` fails if it is stale |
| `Model1-Colab-benchmark-20261006.ipynb` | Ordered phases; each cell bootstraps rclone and verifies the harness against embedded SHA-256 |
| `rig_browser.cjs` | Playwright-over-CDP control of the isolated rig Chrome |
| `../../tests/test_colab_benchmark.py` | Budget, attempt, archive, digest, report, scoped-storage and real-trainer resume tests |

## Storage blast radius

The harness never mounts Drive. Colab's `drive.mount` requires read/write/delete
access to all of Drive plus Photos, contacts and other scopes. Instead, storage
is one folder reached through rclone with Google's `drive.file` scope and the
operator's own OAuth client: the credential can see only files that client
created. `root_folder_id` pins the remote to the benchmark folder.

- The credential is the rclone config section `[bench]` (`type = drive`,
  `scope = drive.file`, `root_folder_id`, client and token). It is stored as the
  Colab secret `TRUESKATE_BENCH_RCLONE`, with notebook access granted to this
  notebook only. Locally it lives in an ignored `tmp/` config with mode 600.
- Every cell refuses a config without `scope = drive.file` and a
  `root_folder_id`.
- `persist` uploads each file and compares Drive's own SHA-256 before anything
  that depends on it is written. A checkpoint is verified before
  `latest.json` names it.

Remote layout under the folder: `code/` (harness), `inputs/` (frozen source,
data, Modal reference, input lock), `results/` (ledger, sessions, checkpoints,
evidence) and the notebook.

## Setup (once)

1. Create a Google Cloud project with the Drive API enabled. Add an External
   OAuth consent screen in Testing with yourself as test user and only the
   `drive.file` scope. Create a **Desktop** client and download its JSON.
2. Authorize locally with the pinned rclone (opens a consent page asking only
   for `drive.file`):
   `rclone config create bench drive client_id=… client_secret=… scope=drive.file`.
3. `rclone mkdir bench:<folder>`, then set `root_folder_id` to its ID.
4. Upload `code/`, `inputs/` and the notebook, and check them with
   `rclone check --one-way`.
5. Rebuild the notebook after any harness change:
   `python scripts/colab/build_notebook.py`.

## Operating rules

- Keep the notebook and folder private. Never upload `.env`, Git metadata or
  any credential other than the scoped Colab secret.
- Execute one phase at a time, never Run All. CPU for preparation and ledger
  cells; free T4 with standard RAM for training.
- Budget: 12 cumulative GPU-allocation hours within 72 calendar hours of the
  first allocation, counting setup, idle time and recovery. At most two
  connection attempts per UTC day; failed connections count. No automatic
  reconnects or keep-alives.
- Allocation is charged from VM boot and closed only after deletion is
  observed in Runtime → Manage sessions.
- At epoch 5 the timing phase stops after verifying the full
  model/optimizer/RNG/shuffle checkpoint remotely. Delete the runtime, close
  the ledger on CPU and start the completion phase on a new T4; the harness
  verifies restoration at epoch 6 against the frozen trainer.
- After an interruption: record the observed release on CPU, register the next
  attempt within the cap, and rerun the matching phase. It resumes from the last
  verified checkpoint.
- Never evaluate test, training-set or new-park holdout partitions.

## Rig browser

The operator browser is a temporary Chrome on `training-server` with its own
profile, under `~/trueskate-ai-runtime/tmp/colab-browser-20261006/`. It listens
for DevTools on rig loopback port 9222 only. Reach it through an SSH tunnel:

```bash
ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N \
  -L 127.0.0.1:19222:127.0.0.1:9222 training-server@training-server
cd scripts/colab && npm ci   # once
node rig_browser.cjs '{"kind":"state"}'
```

Before launching Chrome, loading pages or asking for Screen Sharing, check that
`~/trueskate-ai-runtime/tmp/RIG-BUSY-curved-recording` does not exist on the
rig: collection recordings are timing-critical. Google sign-in is done by the
operator over Screen Sharing. Stop Chrome and the tunnel when not in use; the
profile stays on the rig and is never copied.
