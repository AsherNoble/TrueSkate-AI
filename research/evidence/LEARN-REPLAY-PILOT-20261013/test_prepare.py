"""Synthetic checks for the prospective cohort's safety and leakage guards."""
import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("pilot_prepare", Path(__file__).with_name("prepare.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


@pytest.fixture(scope="module")
def plan():
    return prepare.prepare()


def rehash(plan):
    plan["sha256"] = prepare.digest({k: v for k, v in plan.items() if k != "sha256"})
    return plan


def test_deterministic_prospective_cohort_and_legal_payloads(plan):
    assert prepare.prepare() == plan
    summary = prepare.validate(plan)
    assert summary["candidate_count"] == 600
    assert summary["replay_count"] == 200
    assert summary["sessions"] == 12
    assert plan["physical_gate"] == "blocked-unmeasured"
    assert plan["park_context"] is None


def test_rehashed_session_leak_rejected(plan):
    changed = copy.deepcopy(plan)
    row = next(r for r in changed["candidates"] if r["split"] == "test")
    row["split"] = "train"
    with pytest.raises(ValueError, match="whole-session"):
        prepare.validate(rehash(changed))


def test_rehashed_unsafe_or_mutated_command_rejected(plan):
    changed = copy.deepcopy(plan)
    changed["candidates"][0]["payload"]["actions"][0]["actions"][0]["x"] = 0
    with pytest.raises(ValueError, match="compiled command changed"):
        prepare.validate(rehash(changed))


def test_missing_replay_and_train_selection_rejected(plan):
    changed = copy.deepcopy(plan)
    changed["replay"].pop()
    with pytest.raises(ValueError, match="paired replay"):
        prepare.validate(rehash(changed))
    changed = copy.deepcopy(plan)
    changed["selected_test_ids"][0] = next(r["id"] for r in changed["candidates"] if r["split"] == "train")
    with pytest.raises(ValueError, match="test selection"):
        prepare.validate(rehash(changed))


def test_qualification_or_budget_changes_rejected(plan):
    changed = copy.deepcopy(plan)
    changed["qualification"] = True
    with pytest.raises(ValueError, match="research-only"):
        prepare.validate(rehash(changed))
    changed = copy.deepcopy(plan)
    changed["replay_dispatch_limit"] += 1
    with pytest.raises(ValueError, match="budget"):
        prepare.validate(rehash(changed))
