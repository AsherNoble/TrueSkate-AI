"""Fixture checks for the auto-offloader's fail-closed spin provenance gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "ops" / "offload_corpus_to_modal.sh"


def _check(session: Path, threshold: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "REPO": str(session.parent),
            "PY": sys.executable,
            "MIN_SPIN_FRAC": threshold,
            "PROVENANCE_CHECK_ONLY": str(session),
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def _manifest(session: Path, index: int, payload: object) -> None:
    (session / f"segment_{index:05d}.json").write_text(json.dumps(payload))


def _run_offloader(repo: Path, threshold: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "REPO": str(repo),
            "PY": sys.executable,
            "MODAL": str(repo / "must-not-run-modal"),
            "MIN_SPIN_FRAC": threshold,
            "MAX_ROUNDS": "1",
            "QUIESCENT_MIN": "0",
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_disabled_filter_preserves_unknown_session_behavior(tmp_path: Path) -> None:
    session = tmp_path / "legacy_session"
    session.mkdir()

    result = _check(session, "0")

    assert result.returncode == 0, result.stdout + result.stderr


def test_all_manifests_must_meet_threshold(tmp_path: Path) -> None:
    session = tmp_path / "spin_session"
    session.mkdir()
    _manifest(session, 0, {"mix": {"spin_frac": 0.8}})
    _manifest(session, 1, {"mix": {"spin_frac": 0.95}})

    result = _check(session, "0.8")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 manifest(s), minimum spin_frac=0.8" in result.stdout


def test_one_low_manifest_rejects_otherwise_eligible_session(tmp_path: Path) -> None:
    session = tmp_path / "mixed_session"
    session.mkdir()
    _manifest(session, 0, {"mix": {"spin_frac": 0.8}})
    _manifest(session, 1, {"mix": {"spin_frac": 0.79}})

    result = _check(session, "0.8")

    assert result.returncode != 0
    assert "segment_00001.json: spin_frac=0.79 < 0.8" in result.stdout


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"mix": {"spin_frac": 0.79}}, "spin_frac=0.79 < 0.8"),
        ({"mix": {}}, "invalid provenance"),
        ({"mix": {"spin_frac": True}}, "invalid provenance"),
        ({"mix": {"spin_frac": 1.1}}, "invalid provenance"),
    ],
)
def test_unknown_or_low_spin_manifest_fails_closed(
    tmp_path: Path, payload: object, message: str
) -> None:
    session = tmp_path / "ineligible_session"
    session.mkdir()
    _manifest(session, 0, payload)

    result = _check(session, "0.8")

    assert result.returncode != 0
    assert message in result.stdout


def test_missing_or_malformed_manifests_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    missing.mkdir()
    missing_result = _check(missing, "0.8")
    assert missing_result.returncode != 0
    assert "no segment_*.json manifests" in missing_result.stdout

    malformed = tmp_path / "malformed"
    malformed.mkdir()
    (malformed / "segment_00000.json").write_text("not json")
    malformed_result = _check(malformed, "0.8")
    assert malformed_result.returncode != 0
    assert "invalid provenance" in malformed_result.stdout


def test_ineligible_session_is_skipped_without_deletion(tmp_path: Path) -> None:
    session = tmp_path / "data" / "sls_xctest" / "legacy_session"
    session.mkdir(parents=True)
    _manifest(session, 0, {"mix": {"spin_frac": 0.15}})

    result = _run_offloader(tmp_path, "0.8")

    assert result.returncode == 0, result.stdout + result.stderr
    assert session.is_dir()
    assert "spin_frac=0.15 < 0.8; kept local" in result.stdout
    assert "nothing eligible" in result.stdout


def test_invalid_threshold_is_rejected(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()

    result = _check(session, "not-a-number")

    assert result.returncode == 2
    assert "MIN_SPIN_FRAC must be a finite number in [0, 1]" in result.stdout
