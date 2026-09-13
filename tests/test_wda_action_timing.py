import copy

import pytest

from trueskate_ai.collection.wda_action_timing import (
    BOUNDARIES,
    WDAActionTimingCapture,
    validate_action_timing_report,
)


def _report(count=1):
    records = []
    for sequence in range(count):
        record = {
            "sequence": sequence,
            "outcome": "success",
            "session_id": "session-a",
            "missing_ios_callback": False,
            "ios_callback_result": True,
        }
        record.update(
            {
                name: {"monotonic_s": sequence * 10 + index, "epoch_s": 1000 + sequence * 10 + index}
                for index, name in enumerate(BOUNDARIES)
            }
        )
        records.append(record)
    return {
        "schema_version": 1,
        "build_revision": "revision-a",
        "dropped_records": 0,
        "records": records,
    }


def test_report_validation_requires_complete_ordered_records():
    assert len(validate_action_timing_report(
        _report(2), expected_revision="revision-a", expected_count=2
    )) == 2
    for field, value in (
        ("outcome", "error"),
        ("missing_ios_callback", True),
        ("ios_callback_result", False),
        ("session_id", ""),
    ):
        bad = _report()
        bad["records"][0][field] = value
        with pytest.raises(ValueError):
            validate_action_timing_report(
                bad, expected_revision="revision-a", expected_count=1
            )
    bad = _report()
    bad["records"][0]["submitted_to_ios"]["monotonic_s"] = 99
    with pytest.raises(ValueError, match="ordering"):
        validate_action_timing_report(
            bad, expected_revision="revision-a", expected_count=1
        )


def test_report_validation_rejects_revision_count_and_drops():
    report = _report()
    with pytest.raises(ValueError, match="revision"):
        validate_action_timing_report(
            report, expected_revision="wrong", expected_count=1
        )
    with pytest.raises(ValueError, match="expected 2"):
        validate_action_timing_report(
            report, expected_revision="revision-a", expected_count=2
        )
    bad = copy.deepcopy(report)
    bad["dropped_records"] = 1
    with pytest.raises(ValueError, match="overflowed"):
        validate_action_timing_report(
            bad, expected_revision="revision-a", expected_count=1
        )


def test_capture_enables_and_disables_the_existing_session():
    calls = []

    def request(url, payload=None):
        calls.append((url, payload))
        if url.endswith("/status"):
            return {"sessionId": "a session"}
        if payload is None:
            return {"value": {"schema_version": 1, "build_revision": "revision-a"}}
        return {"value": {"enabled": bool(payload["enabled"]), "records": []}}

    capture = WDAActionTimingCapture(
        wda_port=8103, expected_revision="revision-a", request_json=request
    )
    capture.start()
    assert capture.active
    assert "%20" in calls[1][0]
    assert capture.stop()["records"] == []
    assert not capture.active
