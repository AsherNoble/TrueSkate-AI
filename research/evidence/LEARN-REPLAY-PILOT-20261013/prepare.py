"""Freeze prospective diagnostic commands and replay order; never operates a device.

Run from the repository root using the shared .venv. The park/settings binding,
physical repeatability evidence, media admission, training and replay are separate
dependencies. This plan cannot be consumed by the production collection ledger.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from trueskate_ai.model1.curved.contract import Gesture, digest, save_json
from trueskate_ai.model1.curved.sampling import sample_gesture

SEED = 20261013
KIND = "curved-diagnostic-pilot-plan-v1"


def prepare(seed=SEED):
    """Twelve whole sessions: 400 train /100 validation /100 test commands."""
    splits = ["train"] * 8 + ["validation"] * 2 + ["test"] * 2
    random.Random(seed).shuffle(splits)
    rows = []
    for session_index, split in enumerate(splits):
        session = f"pilot-{session_index:02}"
        for slot in range(50):
            candidate_id = 50 * session_index + slot + 1
            sampler = ("random_chain", "arc")[slot % 2]
            gesture = sample_gesture(seed + candidate_id * 1009, sampler)
            rows.append(dict(id=candidate_id, session=session, split=split,
                             sampler=sampler, gesture=gesture.to_dict(),
                             payload=gesture.payload()))
    # Chosen before any video, labels, model prediction or executor outcome.
    test_ids = [row["id"] for row in rows if row["split"] == "test"]
    selected = sorted(random.Random(seed + 1).sample(test_ids, 20))
    replay = [dict(candidate_id=i, condition=condition, repeat=repeat)
              for i in selected for condition in ("original", "predicted")
              for repeat in range(5)]
    random.Random(seed + 2).shuffle(replay)
    for ordinal, job in enumerate(replay, 1):
        job["dispatch_id"] = ordinal
    plan = dict(kind=KIND, seed=seed, physical_gate="blocked-unmeasured",
                park_context=None, qualification=False,
                candidate_dispatch_limit=600, replay_dispatch_limit=200,
                capture_support_limit=360, replay_support_limit=120,
                candidates=rows, selected_test_ids=selected, replay=replay)
    plan["sha256"] = digest(plan)
    validate(plan)
    return plan


def validate(plan):
    """Fail closed on command mutation, partition leakage and budget changes."""
    if plan.get("kind") != KIND or plan.get("qualification") is not False:
        raise ValueError("research-only plan required")
    if plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("plan digest changed")
    rows = plan["candidates"]
    if (len(rows) != 600 or plan.get("candidate_dispatch_limit") != 600
            or plan.get("replay_dispatch_limit") != 200
            or plan.get("capture_support_limit") != 360
            or plan.get("replay_support_limit") != 120):
        raise ValueError("frozen diagnostic budget changed")
    if {r["id"] for r in rows} != set(range(1, 601)):
        raise ValueError("candidate identities changed")
    groups = {}
    for row in rows:
        gesture = Gesture.from_dict(row["gesture"])
        if gesture.sampler != row["sampler"] or gesture.payload() != row["payload"]:
            raise ValueError("compiled command changed")
        groups.setdefault(row["session"], []).append(row)
    if set(groups) != {f"pilot-{i:02}" for i in range(12)}:
        raise ValueError("session identities changed")
    for items in groups.values():
        if len(items) != 50 or len({r["split"] for r in items}) != 1:
            raise ValueError("whole-session partition required")
        if any(sum(r["sampler"] == s for r in items) != 25 for s in ("arc", "random_chain")):
            raise ValueError("session sampler balance changed")
    if any(sum(r["split"] == s for r in rows) != n
           for s, n in (("train", 400), ("validation", 100), ("test", 100))):
        raise ValueError("partition sizes changed")
    test_ids = {r["id"] for r in rows if r["split"] == "test"}
    selected = plan["selected_test_ids"]
    if len(selected) != 20 or len(set(selected)) != 20 or not set(selected) <= test_ids:
        raise ValueError("prospective test selection changed")
    expected = {(i, c, r) for i in selected for c in ("original", "predicted") for r in range(5)}
    jobs = plan["replay"]
    if (len(jobs) != 200 or {j["dispatch_id"] for j in jobs} != set(range(1, 201))
            or {(j["candidate_id"], j["condition"], j["repeat"]) for j in jobs} != expected):
        raise ValueError("paired replay cohort changed")
    return dict(candidate_count=len(rows), replay_count=len(jobs),
                sessions=len(groups), selected_test_ids=selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        plan = prepare()
        save_json(args.path, plan)  # Exclusive: never overwrite a frozen plan.
    else:
        plan = json.loads(args.path.read_text())
    print(json.dumps(dict(sha256=plan["sha256"], **validate(plan)), indent=2))


if __name__ == "__main__":
    main()
