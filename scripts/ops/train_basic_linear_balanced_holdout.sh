#!/bin/bash
# Train the predeclared MVP-2 device-balanced fresh holdout protocol.
#
# The caller supplies the two completed earlier corpora as train-only legacy
# sources and the newly collected per-device-balanced corpus as `fresh`.  Each
# seed saves its validation-selected checkpoint without opening the fresh test;
# the final Modal call validation-selects a compact ensemble and evaluates test
# exactly once.
set -eu

LEGACY_2K="${1:?usage: train_basic_linear_balanced_holdout.sh LEGACY_2K LEGACY_FRESH FRESH RUN_LABEL [per_device_target]}"
LEGACY_FRESH="${2:?usage: train_basic_linear_balanced_holdout.sh LEGACY_2K LEGACY_FRESH FRESH RUN_LABEL [per_device_target]}"
FRESH="${3:?usage: train_basic_linear_balanced_holdout.sh LEGACY_2K LEGACY_FRESH FRESH RUN_LABEL [per_device_target]}"
RUN_LABEL="${4:?usage: train_basic_linear_balanced_holdout.sh LEGACY_2K LEGACY_FRESH FRESH RUN_LABEL [per_device_target]}"
PER_DEVICE_TARGET="${5:-500}"
REPO=/Users/training-server/trueskate-ai
VOLUME="${MODAL_CORPUS_VOLUME:?set MODAL_CORPUS_VOLUME to a new dedicated Modal volume}"
DATA_SUBDIR="${BASIC_LINEAR_BALANCED_SUBDIR:-basic_linear_balanced_fresh_holdout}"

cd "$REPO"
case "$PER_DEVICE_TARGET" in
  ''|*[!0-9]*) echo "per_device_target must be a non-negative integer" >&2; exit 2 ;;
esac

# Menu flags are an exclusion marker, so scan before validating/admitting the
# fresh corpus.  Then insist it remains both command-unique and balanced.
PYTHONPATH=src .venv/bin/python scripts/data/flag_menu_samples.py --data "$FRESH"
PYTHONPATH=src .venv/bin/python - "$FRESH" "$PER_DEVICE_TARGET" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path
from trueskate_ai.vision.basic_linear_dataset import BasicLinearClipDataset

root, target = Path(sys.argv[1]), int(sys.argv[2])
data = BasicLinearClipDataset(root)
if len(data) != len(set(data.command_keys)):
    raise SystemExit("fresh corpus contains duplicate exact commands")
counts = Counter()
for path in data.sample_paths:
    device = json.loads((path / "meta.json").read_text()).get("device")
    if not isinstance(device, str) or not device:
        raise SystemExit(f"missing explicit device provenance: {path}")
    counts[device] += 1
required = {"iPhone_XR", "iPhone_XR2"}
if set(counts) != required or any(counts[device] < target for device in required):
    raise SystemExit(f"need at least {target} strict fresh commands per device; got {dict(counts)}")
print({"accepted": len(data), "by_device": dict(sorted(counts.items())), "stats": data.stats})
PY

# Two separate legacy subtrees avoid a physical merge while preserving all
# 3,040 previous commands as train-only material.  `fresh` must remain a direct
# child because the splitter deliberately rejects ambiguous source layouts.
PYTHONPATH=src .venv/bin/python scripts/cloud/upload_basic_linear_corpus.py \
  --source "$LEGACY_2K" --volume "$VOLUME" --remote-subdir "$DATA_SUBDIR/legacy/verified_2k" \
  --min-samples 1000
PYTHONPATH=src .venv/bin/python scripts/cloud/upload_basic_linear_corpus.py \
  --source "$LEGACY_FRESH" --volume "$VOLUME" --remote-subdir "$DATA_SUBDIR/legacy/first_fresh" \
  --min-samples 1000
PYTHONPATH=src .venv/bin/python scripts/cloud/upload_basic_linear_corpus.py \
  --source "$FRESH" --volume "$VOLUME" --remote-subdir "$DATA_SUBDIR/fresh" \
  --min-samples "$((PER_DEVICE_TARGET * 2))"

checkpoint_names=""
for seed in 0 1 2; do
  label="${RUN_LABEL}_seed${seed}"
  env MODAL_CORPUS_VOLUME="$VOLUME" .venv/bin/modal run \
    scripts/cloud/train_basic_linear_modal.py::train_remote \
    --data-subdir "$DATA_SUBDIR" --run-label "$label" \
    --epochs 40 --batch-size 8 --lr 1e-3 --seed "$seed" --split-seed 0 --base-channels 16 \
    --split-strategy command --temporal-mixer --fresh-holdout-source fresh \
    --fresh-stratify-by-device --no-evaluate-test
  checkpoint_names="${checkpoint_names:+$checkpoint_names,}basic_linear_${label}.pth"
done

env MODAL_CORPUS_VOLUME="$VOLUME" .venv/bin/modal run \
  scripts/cloud/train_basic_linear_modal.py::evaluate_checkpoint_ensemble \
  --data-subdir "$DATA_SUBDIR" --checkpoint-names "$checkpoint_names" \
  --seed 0 --batch-size 8 --fresh-holdout-source fresh --fresh-stratify-by-device
