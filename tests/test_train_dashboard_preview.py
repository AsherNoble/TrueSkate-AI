"""Dashboard preview discovery: across corpora, and over mp4-packed samples.

Guards the two mismatches that left the rig dashboard permanently STALE while
collection was healthy: the preview pinned to a single corpus dir, and the
aligner packing (and deleting) the PNGs into frames.mp4.
"""
import importlib.util
import json
import time
from pathlib import Path

import pytest

_DASH = Path(__file__).resolve().parent.parent / "scripts" / "train_dashboard.py"
_spec = importlib.util.spec_from_file_location("train_dashboard", _DASH)
dash = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dash)

DEVICE = "iPhone_XR"
PARK = "The Workshop"


def _write_session(root: Path, corpus: str, session: str, gestures: list[dict],
                   *, media: str = "mp4", use_spin: bool = False) -> Path:
    """Build one aligned session dir the way collect+align leave it on disk."""
    sess = root / corpus / session
    sess.mkdir(parents=True)
    manifest = {"device": DEVICE, "park": PARK, "use_spin": use_spin,
                "segment_index": 0, "gestures": gestures}
    (sess / "segment_00000.json").write_text(json.dumps(manifest))
    (sess / "segment_00000.aligned").write_text("")
    for g in gestures:
        sample = sess / dash._park_tag(PARK) / f"sample_{g['gesture_index']:06d}"
        sample.mkdir(parents=True)
        if media == "mp4":
            (sample / "frames.mp4").write_bytes(b"\x00fake-h264")
        elif media == "png":
            (sample / "frame_000.png").write_bytes(b"\x89PNG-a")
            (sample / "frame_001.png").write_bytes(b"\x89PNG-b")
    return sess


def _gestures(n: int, t0: float) -> list[dict]:
    return [{"gesture_index": i, "park": PARK, "t_call_end_epoch_s": t0 + i} for i in range(n)]


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    now = time.time()
    _write_session(tmp_path, "sls_xctest", f"{DEVICE}_20260806_010000", _gestures(3, now - 86400))
    _write_session(tmp_path, "basic_linear_xctest_4k_verified",
                   f"{DEVICE}_20260813_064455", _gestures(4, now - 60))
    return tmp_path


def test_sessions_span_sibling_corpora_newest_first(corpus: Path):
    found = dash._sessions(corpus, DEVICE)
    assert [p.name for p in found] == [f"{DEVICE}_20260813_064455", f"{DEVICE}_20260806_010000"]


def test_sessions_also_works_when_root_is_one_corpus(corpus: Path):
    found = dash._sessions(corpus / "sls_xctest", DEVICE)
    assert [p.name for p in found] == [f"{DEVICE}_20260806_010000"]


def test_preview_serves_newest_mp4_clip(corpus: Path):
    info = dash._latest_preview_clip(corpus, DEVICE)
    assert info["media"] == "video"
    assert info["ctype"] == "video/mp4"
    assert info["path"].name == "frames.mp4"
    assert info["corpus"] == "basic_linear_xctest_4k_verified"
    assert info["path"].parent.name == "sample_000003"  # newest gesture in the manifest


def test_preview_falls_back_to_png_sequence(tmp_path: Path):
    _write_session(tmp_path, "sls_xctest", f"{DEVICE}_20260806_010000",
                   _gestures(2, time.time()), media="png")
    info = dash._latest_preview_clip(tmp_path, DEVICE)
    assert info["media"] == "image"
    assert info["path"].name == "frame_001.png"  # last frame of the sequence


def test_preview_skips_menu_flagged_and_empty_samples(corpus: Path):
    sess = corpus / "basic_linear_xctest_4k_verified" / f"{DEVICE}_20260813_064455"
    park = sess / dash._park_tag(PARK)
    (park / "sample_000003" / ".menu").write_text("")
    for f in (park / "sample_000002").iterdir():
        f.unlink()
    info = dash._latest_preview_clip(corpus, DEVICE)
    assert info["path"].parent.name == "sample_000001"


def test_preview_none_when_no_footage(tmp_path: Path):
    assert dash._latest_preview_clip(tmp_path, DEVICE) is None


def test_collection_status_counts_only_the_recent_window(corpus: Path):
    s = dash._collection_status(corpus, DEVICE)
    assert s["mode"] == "collect"
    assert s["corpus"] == "basic_linear_xctest_4k_verified"
    assert s["session"] == f"{DEVICE}_20260813_064455"
    assert s["park"] == PARK
    assert s["samples_1h"] == 4          # the day-old session is outside the window
    assert s["session_samples"] == 4
    assert s["last_sample_age_s"] < 120


def test_collection_status_reports_the_newest_session_with_a_manifest(corpus: Path):
    """The newest session dir is normally the in-flight one — empty until its
    segment finishes recording — so stats must come from the newest one that
    actually has a manifest, not from session 0."""
    (corpus / "basic_linear_xctest_4k_verified" / f"{DEVICE}_20260813_070409").mkdir()
    s = dash._collection_status(corpus, DEVICE)
    assert s["session"] == f"{DEVICE}_20260813_064455"
    assert s["session_samples"] == 4


def test_collection_status_ignores_calibration_sidecars(corpus: Path):
    """segment_00000.calibration_rejected.json matches a naive segment_*.json
    glob and sorts ahead of the real manifest — it has no park or gestures."""
    sess = corpus / "basic_linear_xctest_4k_verified" / f"{DEVICE}_20260813_064455"
    (sess / "segment_00000.calibration_rejected.json").write_text(
        json.dumps({"reason": "mad_too_high"}))
    s = dash._collection_status(corpus, DEVICE)
    assert s["park"] == PARK
    assert s["samples_1h"] == 4


def test_collection_status_without_a_corpus(tmp_path: Path):
    s = dash._collection_status(tmp_path, DEVICE)
    assert s["mode"] == "collect"
    assert s["note"]


def test_device_status_falls_back_to_collection(tmp_path: Path, corpus: Path):
    s = dash._device_status(tmp_path / "no-logs", DEVICE, corpus)
    assert s["mode"] == "collect"
    assert s["samples_1h"] == 4
