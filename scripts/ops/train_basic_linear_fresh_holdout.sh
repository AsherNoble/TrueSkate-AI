#!/bin/bash
# Upload a legacy corpus and an independently collected corpus into separate
# subtrees, then evaluate only commands from the latter.  This avoids copying
# clips locally and fails closed if any exact gesture command overlaps.
set -eu

LEGACY_OUT="${1:?usage: train_basic_linear_fresh_holdout.sh LEGACY_OUT FRESH_OUT RUN_LABEL [target]}"
FRESH_OUT="${2:?usage: train_basic_linear_fresh_holdout.sh LEGACY_OUT FRESH_OUT RUN_LABEL [target]}"
RUN_LABEL="${3:?usage: train_basic_linear_fresh_holdout.sh LEGACY_OUT FRESH_OUT RUN_LABEL [target]}"
TARGET="${4:-1000}"
REPO=/Users/training-server/trueskate-ai
VOLUME="${MODAL_CORPUS_VOLUME:?set MODAL_CORPUS_VOLUME to a new dedicated Modal volume}"
DATA_SUBDIR="${BASIC_LINEAR_MIXED_SUBDIR:-basic_linear_mixed_fresh_holdout}"

cd "$REPO"
PYTHONPATH=src .venv/bin/python scripts/data/flag_menu_samples.py --data "$FRESH_OUT"
PYTHONPATH=src .venv/bin/python scripts/cloud/upload_basic_linear_corpus.py \
  --source "$LEGACY_OUT" --volume "$VOLUME" --remote-subdir "$DATA_SUBDIR/legacy" \
  --min-samples 1000
PYTHONPATH=src .venv/bin/python scripts/cloud/upload_basic_linear_corpus.py \
  --source "$FRESH_OUT" --volume "$VOLUME" --remote-subdir "$DATA_SUBDIR/fresh" \
  --min-samples "$TARGET"
env MODAL_CORPUS_VOLUME="$VOLUME" .venv/bin/modal run scripts/cloud/train_basic_linear_modal.py \
  --data-subdir "$DATA_SUBDIR" --run-label "$RUN_LABEL" \
  --epochs 40 --batch-size 8 --lr 1e-3 --seed 0 --split-seed 0 --base-channels 16 \
  --split-strategy command --temporal-mixer --fresh-holdout-source fresh
