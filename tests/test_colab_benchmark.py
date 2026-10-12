import importlib.util
import json
from pathlib import Path
import tarfile

import pytest
import torch

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts/colab/benchmark.py")
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def test_allocation_includes_idle_setup_and_interrupted_sessions():
    ledger = {"sessions": [{"allocated_at": 100, "released_at": 3700}, {"allocated_at": 4000}]}
    assert b.remaining(ledger, 7600) == 10 * 3600


def test_calendar_expiry_counts_gaps_and_open_sessions_fail_closed():
    ledger = {"sessions": [{"allocated_at": 100, "released_at": 200}]}
    assert b.remaining(ledger, 100+72*3600) == 0
    ledger = {"sessions": [{"allocated_at": 100}]}
    with pytest.raises(RuntimeError, match="previous allocation"):
        b.register_attempt(ledger, 200)


def test_daily_attempt_cap_has_no_retry_loop():
    ledger = {}
    b.register_attempt(ledger, 100)
    b.register_attempt(ledger, 200)
    with pytest.raises(RuntimeError, match="two connection"):
        b.register_attempt(ledger, 300)
    b.register_attempt(ledger, 86401)


def test_verified_copy_and_archive_traversal(tmp_path):
    source = tmp_path / "input.pth"
    source.write_bytes(b"complete checkpoint")
    record = b.copy_verified(source, tmp_path / "private/checkpoint.pth")
    assert record["sha256"] == b.sha256(source)
    with tarfile.open(tmp_path / "bad.tar", "w") as t:
        t.add(source, arcname="../escaped")
    with pytest.raises(ValueError, match="unsafe"):
        b.safe_extract(tmp_path / "bad.tar", tmp_path / "staged")
    assert not (tmp_path / "escaped").exists()


def test_state_digest_checks_tensor_and_optimizer_content():
    original = {"state": {0: {"step":torch.tensor(5), "moment":torch.tensor([1.,2.])}},
                "param_groups": [{"lr": 0.001}]}
    assert b.state_digest(original) == b.state_digest(original)
    changed = {"state": {0: {"step":torch.tensor(5), "moment":torch.tensor([1.,3.])}},
               "param_groups": [{"lr": 0.001}]}
    assert b.state_digest(original) != b.state_digest(changed)


def test_full_curve_without_recovery_is_incomplete(tmp_path):
    b.write_json(tmp_path / "history.json", {"validation_curve":[.91]*40,
        "epoch_history":[{"seconds":100}]*40})
    b.write_json(tmp_path / "final/verified.json", {})
    ref = {"validation_plateau_mean_last10":.9092111959287532,
           "epoch_history":[{"seconds":265}]*40}
    assert b.report(tmp_path, ref)["classification"] == "incomplete"


def test_quality_rule_and_partial_work_cash_reporting(tmp_path):
    ref = {"validation_plateau_mean_last10":.9092111959287532,
           "epoch_history":[{"seconds":265}]*40}
    partial = b.report(tmp_path, ref)
    assert partial["modal_same_completed_work_value_usd"] is None
    assert partial["cash_savings_after_credits_usd"] is None
    b.write_json(tmp_path / "history.json", {"validation_curve":[.89]*40,
        "epoch_history":[{"seconds":100}]*40})
    b.write_json(tmp_path / "final/verified.json", {})
    b.write_json(tmp_path / "sessions/session-2/restoration.json", {"passed":True,"resume_at_epoch":6})
    b.write_json(tmp_path / "ledger.json", {"sessions":[
        {"boot_id":"first","allocated_at":100,"released_at":200,"release_verified":True,"outcome":"awaiting_runtime_deletion"},
        {"boot_id":"second","allocated_at":201,"released_at":301,"phase":"completion"}]})
    assert b.report(tmp_path, ref)["classification"] == "completed with a quality concern"


def test_actual_frozen_trainer_restores_epoch5_snapshot_on_synthetic_data(tmp_path, monkeypatch):
    import importlib.util
    root = ROOT
    # Frozen e0424f9 source staged by the local input verification (ignored tmp/).
    source = ROOT / "tmp/colab-benchmark-20261006/local-verification/source"
    if not source.exists():
        pytest.skip("run local content staging verification first")
    tspec = importlib.util.spec_from_file_location("frozen_resume_test",source / "scripts/model1/train_basic_linear_regressor.py")
    t = importlib.util.module_from_spec(tspec); tspec.loader.exec_module(t)
    fixture_spec = importlib.util.spec_from_file_location("synthetic_fixture",root / "tests/test_basic_linear_experiment.py")
    fixtures = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixtures)
    for i in range(12):
        fixtures._write_sample(tmp_path, f"segment_{i}", f"sample_{i}",
                               points=[[.20+i*.01,.35],[.58+i*.01,.55]])
    monkeypatch.setattr(t,"_device",lambda:torch.device("cpu"))
    resume=tmp_path / "resume.pth"
    kwargs=dict(data=tmp_path,epochs=40,batch_size=2,lr=1e-3,seed=0,base_channels=2,
                image_width=16,image_height=36,evaluate_test=False,resume_path=resume,lr_schedule="cosine")
    def stop5():
        if torch.load(resume,map_location="cpu",weights_only=False)["completed_epoch"]==5:
            raise RuntimeError("synthetic deletion boundary")
    with pytest.raises(RuntimeError,match="synthetic deletion"):
        t.train(out=tmp_path/"first.pth",checkpoint_callback=stop5,**kwargs)
    snapshot=torch.load(resume,map_location="cpu",weights_only=False)
    b.recovery_trace(t,snapshot,tmp_path/"restoration.json")
    def stop6():
        raise RuntimeError("synthetic resumed epoch complete")
    with pytest.raises(RuntimeError,match="synthetic resumed"):
        t.train(out=tmp_path/"second.pth",checkpoint_callback=stop6,**kwargs)
    restored=json.loads((tmp_path/"restoration.json").read_text())
    assert restored["passed"] and restored["resume_at_epoch"]==6
    assert restored["actual_next_lrs"]==[t.lr_for_epoch(1e-3,6,40,"cosine")]
