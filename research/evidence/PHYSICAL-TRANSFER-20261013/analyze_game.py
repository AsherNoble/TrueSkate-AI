"""Gate 4 for the game cohort, committed before any game touch or review.

Inputs are the cohort directory written by run_game.py and the review exported from
the temporary review.html. Definitions fixed here:

- Technical admission: both recordings ran without error and all 20 scientific
  requests (candidates 1-20) were admitted as native-frame clips.
- Interpretable: rated "meet", not flagged systemic, with a first-contact mark.
- Rendered-onset offset, in native frames: the marked first-contact frame minus the
  first clip frame at or after the calibrated contact onset (meta onset_s).
- Gate 4 passes only if admission is complete, every family has at least 4/5
  interpretable clips, and the offset range over all interpretable clips is at most
  one native frame. Per-family ranges are reported, not gated.
"""
import argparse
import json
from pathlib import Path

import numpy as np

REVIEW_ROUND = 20261013


def onset_offset_frames(sample, onset_frame):
    times = np.asarray(sample["frame_times_s"])
    predicted = int(np.searchsorted(times, sample["onset_s"], side="left"))
    return onset_frame-predicted


def gate(plan, cohort, review):
    criteria = plan["pilot_gate"]
    summary = json.loads((cohort/"run-summary.json").read_text())
    settled = [json.loads(line) for line in (cohort/"settled.jsonl").read_text().splitlines()] \
        if (cohort/"settled.jsonl").exists() else []
    admitted = sorted(s["candidate_id"] for s in settled if s["admitted"])
    manifest = json.loads((cohort/f"audit-{REVIEW_ROUND}"/"manifest.json").read_text())
    if review.get("manifest_sha256") != manifest["sha256"] or review.get("round") != REVIEW_ROUND:
        raise ValueError("review belongs to a different manifest")
    rows = {r["id"]: r for r in review["samples"]}
    if set(rows) != {s["id"] for s in manifest["samples"]}:
        raise ValueError("review must cover every manifest clip exactly once")
    family = {int(c["request_id"].split("-")[1]): c["path_id"]
              for r in plan["game_recordings"] for c in r["commands"] if c["kind"] == "scientific"}
    clips = []
    for sample in manifest["samples"]:
        row = rows[sample["id"]]
        interpretable = (row.get("rating") == "meet" and not row.get("systematic")
                         and row.get("onset_frame") is not None)
        clips.append(dict(id=sample["id"], path_id=family[sample["id"]], rating=row.get("rating"),
                          systematic=row.get("systematic"), reason=row.get("reason", ""),
                          interpretable=interpretable,
                          onset_frame=row.get("onset_frame"),
                          onset_offset_frames=onset_offset_frames(sample, row["onset_frame"])
                          if interpretable else None))
    failures = []
    if summary.get("error") or admitted != list(range(1, 21)):
        failures.append(f"technical admission incomplete: {len(admitted)}/20 admitted, error={summary.get('error')}")
    families = {}
    for name in sorted(set(family.values())):
        own = [c for c in clips if c["path_id"] == name]
        offsets = [c["onset_offset_frames"] for c in own if c["interpretable"]]
        families[name] = dict(clips=len(own), interpretable=len(offsets),
                              offset_range_frames=max(offsets)-min(offsets) if offsets else None)
        if len(offsets) < criteria["minimum_interpretable_rendered_paths_per_family"]:
            failures.append(f"{name}: {len(offsets)}/5 interpretable")
    offsets = [c["onset_offset_frames"] for c in clips if c["interpretable"]]
    offset_range = max(offsets)-min(offsets) if offsets else None
    if offset_range is None or offset_range > criteria["rendered_onset_max_range_native_frames"]:
        failures.append(f"rendered-onset offset range {offset_range} native frames")
    return dict(schema="physical-transfer-game-gate-v1", passed=not failures, failures=failures,
                admitted=admitted, families=families, offset_range_frames=offset_range,
                offset_counts={str(k): offsets.count(k) for k in sorted(set(offsets))}, clips=clips,
                limitations=["rendered start/path observability only; no glow-derived release or hold timing",
                             "reset/settle does not establish identical board state",
                             "five clips per family do not bound rare failures"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path(__file__).resolve().parent/"frozen-plan.json")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = gate(json.loads(args.plan.read_text()), args.cohort, json.loads(args.review.read_text()))
    with args.out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("passed", "failures", "families", "offset_range_frames")}, indent=2))
