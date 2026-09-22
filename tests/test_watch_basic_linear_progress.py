import importlib.util
import json
from pathlib import Path

import pytest


PATH = Path(__file__).parents[1] / "scripts" / "ops" / "watch_basic_linear_progress.py"
SPEC = importlib.util.spec_from_file_location("watch_basic_linear_progress", PATH)
watcher = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(watcher)


def test_milestones_are_exact_ten_percent_boundaries():
    assert watcher.milestones_due(0, 199, 2000) == []
    assert watcher.milestones_due(0, 200, 2000) == [10]
    assert watcher.milestones_due(10, 615, 2000) == [20, 30]
    assert watcher.milestones_due(90, 2500, 2000) == [100]


def test_new_state_treats_baseline_as_already_announced(tmp_path):
    state_path = tmp_path / "xr2.json"
    state = watcher.load_or_initialize_state(
        state_path,
        label="XR2",
        device="iPhone_XR2",
        target=2000,
        baseline=310,
        current=0,
    )
    assert state["last_notified_pct"] == 10
    assert watcher.milestones_due(state["last_notified_pct"], 399, 2000) == []
    assert watcher.milestones_due(state["last_notified_pct"], 400, 2000) == [20]


def test_existing_state_must_match_run_identity(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "label": "XR1", "device": "iPhone_XR", "target": 2000,
        "baseline": 0, "last_notified_pct": 0,
    }))
    with pytest.raises(RuntimeError, match="identity differs"):
        watcher.load_or_initialize_state(
            state_path,
            label="XR2",
            device="iPhone_XR2",
            target=2000,
            baseline=310,
            current=0,
        )


def test_stop_refuses_unrelated_process(monkeypatch, tmp_path):
    monkeypatch.setattr(watcher, "collector_command", lambda _pid: "python unrelated.py")
    with pytest.raises(RuntimeError, match="does not identify"):
        watcher.stop_collector(
            1234, device="iPhone_XR", output=tmp_path.resolve()
        )
