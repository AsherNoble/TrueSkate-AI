"""Freeze the bounded physical-XR transfer diagnostic; performs no device IO."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess

from trueskate_ai.data.control_hitboxes import CONTROL_START_MAP_VERSION
from trueskate_ai.model1.curved.contract import Gesture
from trueskate_ai.model1.curved.collection import (
    SLOTS, CONTROL_SLOTS, RESET_SLOTS, PREPARE_LEAD_S,
)
from trueskate_ai.research.curved_audit import marker_payload, reset_payload


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def make_plan():
    a, m, b = (.35, .45), (.50, .55), (.65, .65)
    specs = [
        ("single_400", (a, b), (0, 400)),
        ("single_800", (a, b), (0, 800)),
        ("split_800", (a, m, b), (0, 400, 800)),
        ("corner_600", ((.35, .40), (.65, .40), (.65, .65)), (0, 300, 600)),
        ("unequal_600", ((.35, .40), (.65, .40), (.65, .65)), (0, 150, 600)),
        ("initial_hold_600", (a, a, b), (0, 200, 600)),
        ("middle_hold_800", (a, m, m, b), (0, 200, 400, 800)),
        ("final_hold_600", (a, b, b), (0, 400, 600)),
    ]
    paths = []
    for name, points, times in specs:
        g = Gesture(points, times, "random_chain", profile=name)
        payload = g.payload()  # checks every quantized segment against controls
        paths.append(dict(path_id=name, gesture=g.to_dict(), payload=payload,
                          payload_sha256=digest(payload)))
    ids = [p[0] for p in specs]
    helper = []
    for repeat in range(1, 11):
        order = ids.copy()
        random.Random(20261013 + repeat).shuffle(order)
        start = len(helper)
        helper.extend(dict(request_id=f"helper-{start+i+1:03}",
                           path_id=name, repeat=repeat) for i, name in enumerate(order))
    game = []
    representative = ["single_800", "corner_600", "unequal_600", "middle_hold_800"]
    for repeat in range(1, 6):
        order = representative.copy()
        random.Random(20261013 + 100 + repeat).shuffle(order)
        for name in order:
            game.append(dict(request_id=f"game-{len(game)+1:03}", path_id=name, repeat=repeat))
    recordings = []
    for index in range(2):
        # Current curved collection schedule includes measured guard/call overhead.
        commands = [dict(kind="control", role=role, slot_s=slot,
                         payload=marker_payload()) for role, slot in
                    CONTROL_SLOTS]
        commands.extend(dict(kind="reset", slot_s=slot, payload=reset_payload())
                        for slot in RESET_SLOTS)
        for request, slot in zip(game[index*10:(index+1)*10],
                                 SLOTS):
            commands.append(dict(kind="scientific", slot_s=slot, **request))
        recordings.append(dict(recording_id=f"game-{index+1:02}",
            pre_roll_reset=dict(kind="reset", payload=reset_payload(), settle_s=3),
            commands=sorted(commands, key=lambda x: x["slot_s"]), stop_s=59,
            maximum_recording_s=60, prepare_lead_s=PREPARE_LEAD_S, guard_lead_s=.25))
    value = dict(schema="physical-transfer-20261013-v1", seed=20261013,
        device="iPhone_XR", device_env_key="IPHONE_XR_UDID", logical_size=[414, 896],
        wda_port=8100, appium_port=4723, control_map=CONTROL_START_MAP_VERSION,
        helper_bundle_id="com.trueskate.research.touchlogger.physical20261013",
        helper_requests=helper, paths=paths, game_recordings=recordings,
        budget=dict(helper_scientific=80, game_scientific=20, calibration=6,
                    resets=6, total_touch_requests=112, automatic_replacements=0),
        current_park=None, current_waypoint=None, current_settings=None,
        source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        schedule_source="src/trueskate_ai/model1/curved/collection.py",
        schedule_source_sha256=hashlib.sha256(Path(
            "src/trueskate_ai/model1/curved/collection.py").read_bytes()).hexdigest(),
        pilot_gate=dict(status="closed-unmeasured", interpretation="pragmatic diagnostic thresholds, not universal label tolerances",
            all_80_complete_uncancelled_no_overflow=True,
            every_path_measured_repeats=10,
            p95_repeat_position_deviation_normalized_max=.01,
            worst_repeat_position_deviation_normalized_max=.03,
            position_comparison="common absolute time from touch B; componentwise median; no duration normalization",
            p95_absolute_release_deviation_from_path_median_ms_max=8.333333333333334,
            required_corner_endpoint_nearest_sample_distance_pt_max=2.,
            required_hold_observed_distinct_timestamp_span_ms_min=50.,
            all_20_game_recordings_technically_admitted=True,
            minimum_interpretable_rendered_paths_per_family=4,
            rendered_onset_max_range_native_frames=1,
            current_context_verified=True,
            unmeasurable_or_any_failed_condition="stop; do not change threshold or replace attempts",
            game_interpretation="rendered start/path observability only; no glow-derived release/input timing or identical-state assertion"),
        training_admission=False, qualification_effect="none",
        live_prerequisites=["separately signed helper without replacing WDA",
            "current device/park/waypoint/settings verification", "exclusive session ownership",
            "healthy unchanged WDA with expected timing revision", "running root tunnel",
            "idle recorder", "foreground/menu/editor and settle guards",
            "native decode and unchanged per-segment calibration"])
    value["sha256"] = digest(value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    plan = make_plan()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        json.dump(plan, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(dict(path=str(args.out), sha256=plan["sha256"], budget=plan["budget"])))
