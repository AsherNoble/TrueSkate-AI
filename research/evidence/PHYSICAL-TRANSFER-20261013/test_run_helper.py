"""Synthetic checks for the helper cohort runner. No device, WDA or rig access."""
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_helper  # noqa: E402
from run_helper import BIND_TOLERANCE_S, bind, dispatch, select_log, verify_plan  # noqa: E402

REVISION = "rev"
BUNDLE = "com.trueskate.research.touchlogger.physical20261013"


@pytest.fixture(scope="module")
def plan():
    return verify_plan(json.loads((HERE/"frozen-plan.json").read_text()))


class Clock:
    def __init__(self):
        self.t = 1000.

    def __call__(self):
        self.t += .01
        return self.t


class FakeTiming:
    """One WDA timing record per request, stamped on the phone's clock."""

    def __init__(self, clock, *, fail_validation_at=None):
        self.clock, self.active, self.count = clock, False, 0
        self.fail_validation_at = fail_validation_at
        self.cleaned = 0

    def start(self):
        assert not self.active
        self.active = True

    def stop(self):
        self.active = False
        self.count += 1
        base = self.clock.t
        stamps = {b: dict(monotonic_s=base+i*.001, epoch_s=base+i*.001)
                  for i, b in enumerate(run_helper_boundaries())}
        record = dict(sequence=0, outcome="success", ios_callback_result=True, missing_ios_callback=False,
                      session_id="s", **stamps)
        if self.count == self.fail_validation_at:
            record["outcome"] = "failure"
        return dict(schema_version=1, build_revision=REVISION, dropped_records=0, records=[record])

    def cleanup(self):
        self.cleaned += 1
        self.active = False


def run_helper_boundaries():
    from trueskate_ai.collection.wda_action_timing import BOUNDARIES
    return BOUNDARIES


def run(plan, tmp_path, *, perform=None, foreground=lambda: BUNDLE, fail_validation_at=None):
    clock = Clock()
    timing = FakeTiming(clock, fail_validation_at=fail_validation_at)
    sent = []

    def default(payload):
        # Spending must already be durable when the RPC is made.
        spent = (tmp_path/"spend.jsonl").read_text().splitlines()
        assert len(spent) == len(sent)+1
        sent.append(payload)
        return {"value": None}

    receipts = dispatch(plan, bundle=BUNDLE, perform=perform or default, foreground=foreground,
                        timing=timing, revision=REVISION, out=tmp_path, clock=clock, epoch=clock,
                        sleep=lambda s: None)
    return receipts, sent, timing


def test_frozen_order_spend_before_rpc_and_receipts(plan, tmp_path):
    receipts, sent, _ = run(plan, tmp_path)
    paths = {p["path_id"]: p["payload"] for p in plan["paths"]}
    assert [r["request_id"] for r in receipts] == [r["request_id"] for r in plan["helper_requests"]]
    assert sent == [paths[r["path_id"]] for r in plan["helper_requests"]]
    assert len((tmp_path/"spend.jsonl").read_text().splitlines()) == 80
    assert len((tmp_path/"receipts.jsonl").read_text().splitlines()) == 80


def test_rpc_error_stops_without_retry(plan, tmp_path):
    calls = []

    def perform(payload):
        calls.append(payload)
        if len(calls) == 3:
            raise TimeoutError("unknown outcome")
        return {"value": None}

    with pytest.raises(TimeoutError):
        run(plan, tmp_path, perform=perform)
    assert len(calls) == 3
    assert len((tmp_path/"spend.jsonl").read_text().splitlines()) == 3  # The unknown one stays spent.
    assert len((tmp_path/"receipts.jsonl").read_text().splitlines()) == 2


def test_foreground_loss_stops_before_spending(plan, tmp_path):
    answers = iter([BUNDLE, BUNDLE, "com.trueaxis.skate"])
    with pytest.raises(RuntimeError, match="not foreground"):
        run(plan, tmp_path, foreground=lambda: next(answers))
    assert len((tmp_path/"spend.jsonl").read_text().splitlines()) == 2


def test_timing_validation_failure_stops(plan, tmp_path):
    with pytest.raises(ValueError, match="failed WDA action"):
        run(plan, tmp_path, fail_validation_at=4)
    assert len((tmp_path/"spend.jsonl").read_text().splitlines()) == 4
    assert len((tmp_path/"receipts.jsonl").read_text().splitlines()) == 3


def test_changed_plan_is_refused(plan):
    changed = json.loads(json.dumps(plan))
    changed["paths"][0]["payload"]["actions"][0]["actions"][0]["x"] += 1
    with pytest.raises(ValueError, match="changed frozen plan"):
        verify_plan(changed)


def receipts_at(submitted):
    return [dict(request_id=f"helper-{i+1:03}", wda=dict(submitted_to_ios=dict(epoch_s=s)))
            for i, s in enumerate(submitted)]


def batches_at(begins, *, boot=100.):
    return {i+1: dict(marker=dict(batch=i+1, reason="release", overflow=False),
                      events=[dict(kind="T", phase="B", ts=b-boot), dict(kind="T", phase="E", ts=b-boot+.4)])
            for i, b in enumerate(begins)}


HEADER = dict(epoch=100., uptime=0.)


def test_bind_accepts_each_contact_inside_its_window():
    bound = bind(receipts_at([10, 12, 14]), HEADER, batches_at([10.03, 12.05, 14.02]))
    assert [b["log_batch"] for b in bound] == [1, 2, 3]
    assert bound[1]["touch_began_minus_submitted_s"] == pytest.approx(.05)


@pytest.mark.parametrize("begins", [
    [10.03, 12.05],                    # Missing contact.
    [10.03, 12.05, 14.02, 15.5],       # Unexplained extra contact.
    [10.03, 11.90, 14.02],             # Second contact began before its request window.
    [10.03, 13.95, 14.02],             # Second contact began in the third request's window.
])
def test_bind_refuses_ambiguous_or_misaligned_batches(begins):
    with pytest.raises(ValueError):
        bind(receipts_at([10, 12, 14]), HEADER, batches_at(begins))


def test_bind_tolerance_is_small():
    assert BIND_TOLERANCE_S <= .05


def write_log(path, epoch, batches):
    rows = [dict(kind="launch", schema="physical-touch-v1", epoch=epoch, uptime=0., w=414, h=896)]
    for number, batch in batches.items():
        rows.append(dict(kind="batch", batch=number, reason="release", overflow=False))
        rows += batch["events"]
    path.write_text("".join(json.dumps(r)+"\n" for r in rows))


def test_select_log_requires_one_launch_with_contacts(tmp_path):
    write_log(tmp_path/"touches-a.jsonl", 90., batches_at([10.03]))
    write_log(tmp_path/"touches-b.jsonl", 80., {})
    path, header, batches = select_log(tmp_path, 100.)
    assert path.name == "touches-a.jsonl" and list(batches) == [1]
    write_log(tmp_path/"touches-c.jsonl", 95., batches_at([11.]))
    with pytest.raises(ValueError, match="one logger launch"):
        select_log(tmp_path, 100.)


def test_select_log_refuses_launch_after_first_request(tmp_path):
    write_log(tmp_path/"touches-a.jsonl", 101., batches_at([10.03]))
    with pytest.raises(ValueError, match="launched after"):
        select_log(tmp_path, 100.)


def fake_devicectl(files, *, fail_copy=()):
    calls = []

    def runner(args, timeout):
        calls.append(args)
        if args[3] == "info":
            out = Path(args[args.index("--json-output")+1])
            out.write_text(json.dumps(dict(result=dict(files=[dict(name=n) for n in files]))))
            return dict(args=args, exit_code=0, stdout="", stderr="")
        source = args[args.index("--source")+1]
        if source.split("/")[-1] in fail_copy:
            return dict(args=args, exit_code=1, stdout="", stderr="timed out")
        Path(args[args.index("--destination")+1]).write_text("{}\n")
        return dict(args=args, exit_code=0, stdout="", stderr="")
    return runner, calls


def test_retrieve_copies_each_touch_log_by_exact_path(tmp_path):
    runner, calls = fake_devicectl(["touches-b.jsonl", "notes.txt", "touches-a.jsonl"])
    records, retrieved = run_helper.retrieve("udid", BUNDLE, tmp_path/"docs", runner=runner)
    assert retrieved
    sources = [c[c.index("--source")+1] for c in calls if c[3] == "copy"]
    assert sources == ["Documents/touches-a.jsonl", "Documents/touches-b.jsonl"]
    assert sorted(p.name for p in (tmp_path/"docs").iterdir()) == ["touches-a.jsonl", "touches-b.jsonl"]
    assert all(r["sha256"] for r in records if r["step"] == "copy")


@pytest.mark.parametrize("files,fail", [([], ()), (["touches-a.jsonl"], ("touches-a.jsonl",))])
def test_retrieve_reports_missing_or_failed_logs(tmp_path, files, fail):
    runner, _ = fake_devicectl(files, fail_copy=fail)
    _, retrieved = run_helper.retrieve("udid", BUNDLE, tmp_path/"docs", runner=runner)
    assert not retrieved
