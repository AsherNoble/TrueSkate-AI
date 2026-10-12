# Free-Colab Model 1 benchmark harness

Runs one frozen Model 1 training seed on a free Colab GPU, with an allocation
ledger, verified per-epoch checkpoints in private Drive and a deliberate
runtime-deletion recovery test. Experiment record:
[M1-COLAB-20261006](../../research/experiments/M1-COLAB-20261006.md).

| File | Role |
|---|---|
| `benchmark.py` | Harness run inside the pinned `.venv`: staging, preflight, training via the frozen trainer's checkpoint callback, restoration check, report |
| `supervisor.py` | Runs in the Colab kernel: Drive mount, ledger, budget deadline, pinned install, runtime release |
| `Model1-Colab-benchmark-20261006.ipynb` | Five ordered phases; embeds `benchmark.py` and `supervisor.py` byte-for-byte |
| `rig_browser.cjs` | Playwright-over-CDP control of the isolated rig Chrome |
| `../../tests/test_colab_benchmark.py` | Synthetic budget, attempt, archive, digest, report and real-trainer resume tests |

The notebook's embedded copies must match these files. Check before uploading:

```bash
shasum -a 256 scripts/colab/benchmark.py scripts/colab/supervisor.py
```

## Operating rules

- Keep the notebook and Drive folder private. Upload only the input bundle
  (frozen source, data, Modal reference checkpoint and history, input lock).
  Never upload credentials, `.env` or Git metadata.
- Execute one phase at a time, never Run All. CPU for Drive authorization,
  preparation and ledger closing; free T4 with standard RAM for training.
- Budget: 12 cumulative GPU-allocation hours within 72 calendar hours of the
  first allocation, counting setup, idle time and recovery. At most two
  connection attempts per UTC day. No automatic reconnects or keep-alives.
- Allocation is charged from VM boot and closed only after deletion is
  observed in Runtime → Manage sessions.
- At epoch 5 the timing phase saves and checksums the full
  model/optimizer/RNG/shuffle checkpoint in Drive, then stops. Delete the
  runtime, close the ledger on CPU and start the completion phase on a new T4;
  the harness verifies restoration at epoch 6 against the frozen trainer.
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

Google sign-in and the first Drive authorization are done by the operator over
Screen Sharing. Stop Chrome and the tunnel when not in use; the profile stays on
the rig and is never copied.
