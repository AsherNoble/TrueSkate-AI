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

# Stand-in for the drive.file rclone remote: "bench:<rel>" maps under FAKE_REMOTE.
# Every operation is logged; FAKE_CORRUPT lists remote paths whose hash lies.
FAKE_RCLONE = r'''
import hashlib, os, shutil, sys
from pathlib import Path
remote = Path(os.environ["FAKE_REMOTE"])
def resolve(arg):
    return (remote / arg[len("bench:"):], arg[len("bench:"):]) if arg.startswith("bench:") else (Path(arg), None)
op, *args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(" ".join([op] + [resolve(a)[1] for a in args if a.startswith("bench:")]) + "\n")
if op == "copyto":
    (src, _), (dst, _) = resolve(args[0]), resolve(args[1])
    dst.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dst)
elif op == "copy":
    (src, _), (dst, _) = resolve(args[0]), resolve(args[1])
    if not src.is_dir():
        sys.stderr.write("error reading source root directory: directory not found\n"); sys.exit(3)
    shutil.copytree(src, dst, dirs_exist_ok=True)
elif op == "hashsum":
    path, rel = resolve(args[1])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if rel in os.environ.get("FAKE_CORRUPT", "").split(","):
        digest = "0" * 64
    print(f"{digest}  {path.name}")
else:
    sys.exit(f"unsupported fake rclone op {op}")
'''


@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    import sys
    script = tmp_path / "fake_rclone.py"
    script.write_text(FAKE_RCLONE)
    wrapper = tmp_path / "rclone"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(0o755)
    remote, store, log = tmp_path / "remote", tmp_path / "store", tmp_path / "rclone.log"
    remote.mkdir(); store.mkdir(); log.touch()
    monkeypatch.setenv("RCLONE_BIN", str(wrapper))
    monkeypatch.setenv("FAKE_REMOTE", str(remote))
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setattr(b, "STORE_ROOT", store)
    return remote, store, log


def test_persist_uploads_and_checks_remote_checksum(fake_remote, monkeypatch):
    remote, store, log = fake_remote
    b.write_json(store / "results/ledger.json", {"sessions": []})
    b.persist(store / "results/ledger.json")
    assert (remote / "results/ledger.json").read_bytes() == (store / "results/ledger.json").read_bytes()
    assert log.read_text().splitlines() == ["copyto results/ledger.json", "hashsum results/ledger.json"]
    monkeypatch.setenv("FAKE_CORRUPT", "results/ledger.json")
    with pytest.raises(RuntimeError, match="remote checksum mismatch"):
        b.persist(store / "results/ledger.json")


def _resume_snapshot(path, epoch):
    torch.save({"completed_epoch": epoch, "validation_curve": [.5] * epoch,
                "epoch_history": [{"seconds": 1.0}] * epoch, "best": {"epoch": epoch}}, path)


def test_checkpoint_is_verified_remotely_before_pointer_names_it(fake_remote, tmp_path, monkeypatch):
    remote, store, log = fake_remote
    _resume_snapshot(tmp_path / "resume.pth", 5)
    record = b.commit_checkpoint(tmp_path / "resume.pth", store / "results")
    assert record["epoch"] == 5 and record["relative_path"] == "checkpoints/epoch-05.pth"
    ops = log.read_text().splitlines()
    assert ops[:3] == ["copyto results/checkpoints/epoch-05.pth", "hashsum results/checkpoints/epoch-05.pth",
                       "copyto results/latest.json"]
    assert json.loads((remote / "results/latest.json").read_text())["sha256"] == \
        b.sha256(remote / "results/checkpoints/epoch-05.pth")
    _resume_snapshot(tmp_path / "resume.pth", 6)
    monkeypatch.setenv("FAKE_CORRUPT", "results/checkpoints/epoch-06.pth")
    with pytest.raises(RuntimeError, match="remote checksum mismatch"):
        b.commit_checkpoint(tmp_path / "resume.pth", store / "results")
    assert json.loads((remote / "results/latest.json").read_text())["epoch"] == 5
    assert json.loads((store / "results/latest.json").read_text())["epoch"] == 5


def test_persist_is_local_only_without_store(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "STORE_ROOT", None)
    monkeypatch.setenv("RCLONE_BIN", str(tmp_path / "missing-rclone"))
    b.write_json(tmp_path / "x.json", {})
    b.persist(tmp_path / "x.json")


def _supervisor():
    sspec = importlib.util.spec_from_file_location("supervisor", ROOT / "scripts/colab/supervisor.py")
    s = importlib.util.module_from_spec(sspec)
    sspec.loader.exec_module(s)
    return s


def test_scoped_ledger_flow_verifies_inputs_and_caps_attempts(fake_remote, monkeypatch):
    remote, store, log = fake_remote
    s = _supervisor()
    (store / "code").mkdir()
    (store / "code/benchmark.py").write_bytes((ROOT / "scripts/colab/benchmark.py").read_bytes())
    files = {"data.tar": b"shards", "reference/modal_seed0.json":
             json.dumps({"validation_plateau_mean_last10": .9092, "epoch_history": []}).encode()}
    for name, data in files.items():
        (remote / "inputs" / name).parent.mkdir(parents=True, exist_ok=True)
        (remote / "inputs" / name).write_bytes(data)
    lock = {"files": {n: {"bytes": len(d), "sha256": __import__("hashlib").sha256(d).hexdigest()}
                      for n, d in files.items()}}
    b.write_json(store / "inputs/input-lock.json", lock)
    s.prepare(store)
    ledger = json.loads((remote / "results/ledger.json").read_text())
    assert len(ledger["connection_attempts"]) == 1 and "prepared_attempt_at" in ledger
    s.record_failed_connection("no GPU available", store)
    ledger = json.loads((remote / "results/ledger.json").read_text())
    assert ledger["connection_attempts"][0]["failed"] and "prepared_attempt_at" not in ledger
    s.next_attempt(store)
    with pytest.raises(RuntimeError, match="two connection"):
        s.next_attempt(store)
    (remote / "inputs/data.tar").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="remote input hash mismatch: data.tar"):
        s.prepare(store)


def test_committed_notebook_matches_harness_and_refuses_broad_credentials():
    nspec = importlib.util.spec_from_file_location("build_notebook", ROOT / "scripts/colab/build_notebook.py")
    n = importlib.util.module_from_spec(nspec)
    nspec.loader.exec_module(n)
    assert n.NOTEBOOK.read_text() == n.render(), "run scripts/colab/build_notebook.py"
    code = ["".join(c["source"]) for c in n.build()["cells"] if c["cell_type"] == "code"]
    for source in code:
        compile(source, "cell", "exec")
        assert "drive.mount" not in source and "'scope = drive.file' not in conf.splitlines()" in source
        assert b.sha256(ROOT / "scripts/colab/benchmark.py") in source


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
