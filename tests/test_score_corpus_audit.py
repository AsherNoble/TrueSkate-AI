import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "score_corpus_audit", Path(__file__).parents[1] / "scripts" / "inspect" / "score_corpus_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def _inputs(frames):
    selection = {"samples": [{"order": i + 1, "source": f"s{i}"} for i in range(len(frames))]}
    expected = [{"source": f"s{i}", "park": "P", "device": "D", "rate": 1.0,
                 "expected_first_trace_frame_0based": 7} for i in range(len(frames))]
    labels = {f"{i:03d}/s{i}": ({"uncertain": True} if f == "u" else {"frame_index_0based": f, "uncertain": False})
              for i, f in enumerate(frames) if f is not None}
    return selection, expected, labels


def test_audit_passes_only_when_every_clip_within_one_frame():
    assert audit.score(*_inputs([7, 8, 6]))["verdict"] == "pass"
    result = audit.score(*_inputs([7, 9, "u", None]))
    assert result["verdict"] == "fail"
    assert result["satisfactory"] == 1 and result["unclear"] == 1 and result["missing"] == 1
    assert result["exact"] == 1 and len(result["failures"]) == 3
