"""Verify preserved research evidence without operating devices or reading movies."""
import ast
import gzip
import hashlib
import json
import re
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_json(path):
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    index = read_json(ROOT / "bundle-sha256.json")
    for name, expected in index.items():
        assert digest(ROOT / name) == expected, name
    counts = {"json": 0, "python": 0}
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json" or path.name.endswith(".json.gz"):
            read_json(path)
            counts["json"] += 1
        elif path.suffix == ".py":
            ast.parse(path.read_text())
            counts["python"] += 1
    provenance = read_json(ROOT / "provenance.json")
    for name, row in provenance["files"].items():
        assert digest(ROOT / name) == row["sha256"], name
    retained = read_json(ROOT / "xctest-transfer/preservation.json")
    for name, row in retained["files"].items():
        assert digest(ROOT / "xctest-transfer" / name) == row["sha256"], name
    score = read_json(ROOT / "review/score.json")
    assert score["counts"] == {"meet": 83, "fail": 16, "unclear": 1}
    assert score["passed"] is False
    main_run = read_json(ROOT / "xctest-transfer/resume-20261013/summary.json.gz")
    assert main_run["request_count"] == 745 and main_run["all_complete"]
    native = read_json(ROOT / "xctest-transfer/boundary-20261013/native-record-validation.json")
    assert native["requests_checked"] == 255
    assert native["all_timestamps_and_positions_match_commands"]
    observer = read_json(ROOT / "xctest-transfer/observer-20261013/summary.json")
    assert sum(group["requests"] for group in observer.values()) == 90
    assert all(group["callbacks_over50ms"] == 0 for group in observer.values())
    request_counts = {}
    for folder, expected in (("resume-20261013",745),("boundary-20261013",255),("observer-20261013",90)):
        path = ROOT / "xctest-transfer" / folder / "requests.jsonl.gz"
        requests = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
        assert len(requests) == expected
        assert all({"B","E"}.issubset({event.get("ph") for event in row["touches"]}) for row in requests)
        request_counts[folder] = len(requests)
    decoder = read_json(ROOT / "decoder-followup/validation.json")
    assert decoder["audit"]["baseline"]["passed"] == 95
    assert decoder["audit"]["refit_grid_coords100_quantized"]["passed"] == 100
    assert decoder["synthetic"]["baseline"]["passed"] == 810
    assert decoder["synthetic"]["refit_grid_coords100_quantized"]["passed"] == 838
    for cohort in ("audit", "synthetic"):
        assert all(not row["compatibility_failures"] for row in decoder[cohort].values())
    arithmetic = read_json(ROOT / "sim-video-followup/verification-preserved.json")
    assert arithmetic["exact_command_count"] == 100
    assert arithmetic["development_and_heldout_recordings_disjoint"]
    assert all(row["match"] for row in arithmetic["arithmetic"].values())
    transfer = read_json(ROOT / "xctest-transfer/resume-20261013/r6-transfer.json.gz")
    errors = []
    for name, row in transfer["commands"].items():
        times = [point[0] for point in row["command"]]
        predicted = sum(math.ceil((b-a)*60/1000-1e-10)*1000/60 for a,b in zip(times,times[1:]))
        observed = main_run["by_command"][name]["observed_duration_ms"]["median"]
        errors.append(abs(predicted-observed))
    compensation = read_json(ROOT / "compensation-followup/results.json")
    expected = compensation["duration_models"]["sum_ceil_segment_ticks"]["round6_holdout"]["mae_ms"]
    assert len(errors) == 100 and abs(sum(errors)/100-expected) < 1e-10
    assert compensation["inverse_easing"]["forward_inverse_max_abs"] < 1e-12
    broken = []
    reports = [ROOT / "README.md", ROOT.parent.parent / "experiments/EXEC-TRANSFER-20261013.md"]
    reports.extend(ROOT.rglob("REPORT.md"))
    for report in reports:
        for target in re.findall(r"\]\(([^\n)]+)\)", report.read_text()):
            if target.startswith(("http:", "https:", "#", "/")):
                continue
            if not (report.parent / target.split("#")[0]).resolve().exists():
                broken.append([str(report), target])
    assert not broken, broken
    result = {
        "preserved_file_hashes_checked": len(index),
        "parsed_files": counts,
        "research_numeric_invariants": "passed",
        "preserved_complete_request_counts": request_counts,
        "local_report_links": "passed",
        "original_source_verification_complete_now": False,
        "limitation": "Original movies/profiles are unavailable. This checks preserved outputs and arithmetic, not raw-source regeneration or physical transfer.",
    }
    (ROOT / "bundle-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
