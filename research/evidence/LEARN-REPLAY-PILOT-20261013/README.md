# Prospective learning/replay pilot

Prepared only. No physical request, pilot collection, training or replay occurred.
The physical helper/signing prerequisite remains blocked. This directory has no
device-control entry point.

- `plan.json.gz`: 600 prospective legal commands, twelve whole sessions,
  preselected 20 test IDs and 200 randomized paired replay jobs.
- `summary.json`: plan and serialized artifact hashes, cohort checks and zero
  execution counts.
- `prepare.py`: deterministic generator/verifier, seed 20261013. All payloads
  pass the existing curved gesture and quantized protected-control compiler.
- `test_prepare.py`: synthetic mutation, partition leakage, budget and paired
  replay checks; five tests passed.

Plan digest: `f06fce86b507533e64b84520fb6c6719f0d9df99bd00c36913390a526bb1be6c`.

From this worktree's repository root, reproduce without changing frozen evidence:

```sh
mkdir -p tmp/learn-replay-verification
/Users/ashernoble/Projects/Robotics\ \&\ hardware/TrueSkate-AI/.venv/bin/python research/evidence/LEARN-REPLAY-PILOT-20261013/prepare.py prepare tmp/learn-replay-verification/plan.json
/Users/ashernoble/Projects/Robotics\ \&\ hardware/TrueSkate-AI/.venv/bin/python research/evidence/LEARN-REPLAY-PILOT-20261013/prepare.py verify tmp/learn-replay-verification/plan.json
/Users/ashernoble/Projects/Robotics\ \&\ hardware/TrueSkate-AI/.venv/bin/python -m pytest -q research/evidence/LEARN-REPLAY-PILOT-20261013/test_prepare.py
```

The prepare command refuses to overwrite an existing output. Do not give it a
previously frozen plan path. Read the
[experiment protocol](../../experiments/LEARN-REPLAY-PILOT-20261013.md) before
binding park context or proposing live operation.
