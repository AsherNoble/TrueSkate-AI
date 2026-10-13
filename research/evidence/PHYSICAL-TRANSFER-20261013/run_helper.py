"""Bounded XR1 helper cohort: the 80 frozen requests to the separately signed touch logger.

Runs on training-server from a staged checkout with the runtime .venv. One W3C
request per contact, in frozen order, with spending made durable before every RPC.
The first error, foreground loss or timing-validation failure stops the cohort; no
request is retried or replaced. Never restarts WDA, starts the recorder or touches
True Skate beyond activating it again at the end.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
TRUE_SKATE = "com.trueaxis.skate"
# Only if Xcode refused the frozen identifier for the free team (operator-reported).
FALLBACK_BUNDLE = "com.ashernoble.touchlogger.physical20261013"
ENV_FILE = Path("/Users/training-server/trueskate-ai/.env")
RELEASE = Path("/Users/training-server/trueskate-ai")
LOCK = Path("/Users/training-server/trueskate-ai-runtime/tmp/RIG-BUSY-curved-recording")
GAP_S = 1.0             # Fixed idle gap after each request returns; contacts never overlap.
BIND_TOLERANCE_S = .05  # Both stamps come from the phone's own clock.


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def verify_plan(plan):
    if plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("changed frozen plan")
    if len(plan["helper_requests"]) != plan["budget"]["helper_scientific"]:
        raise ValueError("helper cohort does not match its frozen budget")
    return plan


def append_durable(path, row):
    with open(path, "a") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str)+"\n")
        handle.flush()
        os.fsync(handle.fileno())


def dispatch(plan, *, bundle, perform, foreground, timing, revision, out,
             clock=time.monotonic, epoch=time.time, sleep=time.sleep, gap_s=GAP_S):
    """Send each frozen helper request once. Returns receipts; raises on the first failure."""
    from trueskate_ai.collection.wda_action_timing import validate_action_timing_report
    paths = {p["path_id"]: p for p in plan["paths"]}
    receipts = []
    for request in plan["helper_requests"]:
        path = paths[request["path_id"]]
        payload = path["payload"]
        if digest(payload) != path["payload_sha256"]:
            raise ValueError("frozen payload changed: "+request["path_id"])
        if foreground() != bundle:
            raise RuntimeError("helper not foreground before "+request["request_id"])
        # Durable before the RPC: an unknown outcome is spent, never repeated.
        append_durable(out/"spend.jsonl", dict(request_id=request["request_id"], path_id=request["path_id"],
                                                payload_sha256=path["payload_sha256"],
                                                spent_epoch_s=epoch(), spent_monotonic_s=clock()))
        timing.start()
        try:
            started = dict(monotonic_s=clock(), epoch_s=epoch())
            response = perform(payload)
            returned = dict(monotonic_s=clock(), epoch_s=epoch())
        except BaseException:
            timing.cleanup()
            raise
        record = validate_action_timing_report(timing.stop(), expected_revision=revision, expected_count=1)[0]
        receipt = dict(request, payload_sha256=path["payload_sha256"], host_call_start=started,
                       host_call_return=returned, response=response, wda=record)
        append_durable(out/"receipts.jsonl", receipt)
        receipts.append(receipt)
        sleep(gap_s)
    return receipts


def bind(receipts, header, batches, *, tolerance_s=BIND_TOLERANCE_S):
    """Bind batch k to request k only if that contact began inside its own request window."""
    order = sorted(batches)
    if len(order) != len(receipts):
        raise ValueError(f"{len(order)} logger batches for {len(receipts)} completed requests")
    boot_epoch = header["epoch"]-header["uptime"]
    bound = []
    for k, (receipt, number) in enumerate(zip(receipts, order)):
        begins = [e for e in batches[number]["events"] if e.get("kind") == "T" and e.get("phase") == "B"]
        if len(begins) != 1:
            raise ValueError(f"batch {number} lacks exactly one touch begin")
        began = boot_epoch+begins[0]["ts"]
        opened = receipt["wda"]["submitted_to_ios"]["epoch_s"]
        closes = (receipts[k+1]["wda"]["submitted_to_ios"]["epoch_s"] if k+1 < len(receipts)
                  else float("inf"))
        if not opened-tolerance_s <= began < closes-tolerance_s:
            raise ValueError(f"batch {number} began outside {receipt['request_id']}'s request window")
        bound.append(dict(receipt, log_batch=number, touch_began_minus_submitted_s=began-opened))
    return bound


def select_log(directory, first_submission_epoch):
    """The one launch log holding contacts; contacts in any other log are unexplained."""
    sys.path.insert(0, str(HERE))
    from analyze import read_batches
    logs = []
    for path in sorted(Path(directory).rglob("touches-*.jsonl")):
        header, batches = read_batches(path)
        if batches:
            logs.append((path, header, batches))
    if len(logs) != 1:
        raise ValueError(f"expected one logger launch with contacts; found {len(logs)}")
    path, header, batches = logs[0]
    if header["epoch"] > first_submission_epoch:
        raise ValueError("logger launched after the first request")
    return path, header, batches


def run(args, *, timeout=60):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return dict(args=args, exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)


def wda(path, port):
    with urlopen(f"http://127.0.0.1:{port}{path}", timeout=5) as response:
        return json.load(response)


def rig_state(port, udid):
    """Read-only state snapshot for preflight/postflight evidence."""
    disabled = run(["launchctl", "print-disabled", "gui/501"])["stdout"]
    return dict(utc=datetime.now(timezone.utc).isoformat(),
                wda_status=wda("/status", port), foreground=wda("/wda/activeAppInfo", port),
                recorder=wda("/wda/video", port),
                tunnel_running="state = running" in run(
                    ["launchctl", "print", "system/com.trueskate.remotexpc-tunnel"])["stdout"],
                collector_and_watchdog_disabled_lines=[line.strip() for line in disabled.splitlines()
                    if "com.trueskate.collect" in line or "com.trueskate.watchdog" in line],
                release=str(RELEASE.resolve()),
                release_sha=run(["git", "-C", str(RELEASE), "rev-parse", "HEAD"])["stdout"].strip(),
                xr1_udid_sha256=hashlib.sha256(udid.encode()).hexdigest())


def installed(udid, bundle, out):
    listing = out/"installed-app.json"
    result = run(["xcrun", "devicectl", "device", "info", "apps", "--device", udid,
                  "--bundle-id", bundle, "--json-output", str(listing)], timeout=120)
    apps = json.loads(listing.read_text())["result"]["apps"] if not result["exit_code"] else []
    return [a for a in apps if a.get("bundleIdentifier") == bundle]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=HERE/"frozen-plan.json")
    parser.add_argument("--sign-receipt", type=Path, required=True)
    parser.add_argument("--wda-revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if "training-server" not in socket.gethostname():
        raise SystemExit("the helper cohort runs on training-server")
    plan = verify_plan(json.loads(args.plan.read_text()))
    sign = json.loads(args.sign_receipt.read_text())
    bundle = sign["bundle_id"]
    if bundle not in (plan["helper_bundle_id"], FALLBACK_BUNDLE):
        raise SystemExit("signed helper bundle is not the frozen helper")
    args.out.mkdir(parents=True, exist_ok=False)
    descriptor = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)  # Refuses a busy rig.
    with os.fdopen(descriptor, "w") as handle:
        json.dump(dict(owner="physical-transfer helper cohort", pid=os.getpid(),
                       started=datetime.now(timezone.utc).isoformat(), out=str(args.out)), handle)
    error = None
    receipts = []
    try:
        from dotenv import load_dotenv
        load_dotenv(ENV_FILE, override=False)
        from trueskate_ai.sim.device import DeviceSession, DEVICES
        from trueskate_ai.collection.wda_action_timing import WDAActionTimingCapture
        from trueskate_ai.model1.curved.collection import attachment_guard
        cfg = next(d for d in DEVICES if d["name"] == plan["device"])
        udid = os.environ[plan["device_env_key"]]
        port = plan["wda_port"]
        state = rig_state(port, udid)
        (args.out/"preflight.json").write_text(json.dumps(state, indent=2)+"\n")
        if (not state["tunnel_running"] or state["recorder"].get("value") is not None
                or state["foreground"].get("value", {}).get("bundleId") != TRUE_SKATE):
            raise RuntimeError("preflight failed: tunnel, idle recorder and True Skate foreground required")
        attachment_guard(udid)
        app = installed(udid, bundle, args.out)
        if not app:
            raise RuntimeError("signed helper is not installed on XR1")

        def foreground():
            return wda("/wda/activeAppInfo", port).get("value", {}).get("bundleId")

        worker = DeviceSession(cfg, new_command_timeout_s=900)
        worker.connect()
        try:
            driver = worker.driver
            driver.execute_script("mobile: activateApp", {"bundleId": bundle})
            time.sleep(2)
            if foreground() != bundle:
                raise RuntimeError("helper did not come to the foreground (untrusted developer?)")
            timing = WDAActionTimingCapture(wda_port=port, expected_revision=args.wda_revision)
            try:
                receipts = dispatch(plan, bundle=bundle, foreground=foreground, timing=timing,
                                    revision=args.wda_revision, out=args.out,
                                    perform=lambda payload: driver.execute("actions", payload))
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
                print("HELPER_DISPATCH_STOPPED "+error, flush=True)
            finally:
                driver.execute_script("mobile: activateApp", {"bundleId": TRUE_SKATE})
                time.sleep(2)
        finally:
            worker.disconnect()
        if error:  # The completed receipts are still analysed; the gate cannot pass.
            receipts = [json.loads(line) for line in (args.out/"receipts.jsonl").read_text().splitlines()] \
                if (args.out/"receipts.jsonl").exists() else []
        copy = run(["xcrun", "devicectl", "device", "copy", "from", "--device", udid,
                    "--domain-type", "appDataContainer", "--domain-identifier", bundle,
                    "--source", "Documents", "--destination", str(args.out/"device-documents")], timeout=300)
        (args.out/"retrieval.json").write_text(json.dumps(copy, indent=2)+"\n")
        result = dict(error=error, spent=len((args.out/"spend.jsonl").read_text().splitlines())
                      if (args.out/"spend.jsonl").exists() else 0, completed=len(receipts))
        if copy["exit_code"]:
            result["analysis_error"] = "logger retrieval failed"
        elif receipts:
            sys.path.insert(0, str(HERE))
            from analyze import analyze
            try:
                log, header, batches = select_log(args.out/"device-documents",
                                                  receipts[0]["wda"]["submitted_to_ios"]["epoch_s"])
                bound = bind(receipts, header, batches)
                (args.out/"receipts.json").write_text(json.dumps(bound, indent=2, default=str)+"\n")
                analysis = analyze(plan, bound, header, batches)
                (args.out/"helper-result.json").write_text(json.dumps(analysis, indent=2)+"\n")
                result.update(log=str(log), gate=analysis["helper_gate"])
            except ValueError as exc:
                result["analysis_error"] = str(exc)
        postflight = rig_state(port, udid)
        attachment_guard(udid)
        (args.out/"postflight.json").write_text(json.dumps(postflight, indent=2)+"\n")
        (args.out/"run-summary.json").write_text(json.dumps(result, indent=2)+"\n")
        passed = bool(result.get("gate", {}).get("passed")) and not error
        print(f"HELPER_COHORT_DONE gate={'passed' if passed else 'failed'} "
              f"spent={result['spent']} completed={result['completed']}", flush=True)
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
