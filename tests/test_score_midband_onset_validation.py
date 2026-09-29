import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "inspect" / "score_midband_onset_validation.py"
    spec = importlib.util.spec_from_file_location("test_score_midband", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(mid_early_frames, ordinary_early_frames):
    samples, predictions, labels = [], [], {}
    order = 0
    for category, frames in (("mid_high", mid_early_frames), ("ordinary", ordinary_early_frames)):
        for index, frame in enumerate(frames):
            for role, labelled in (("early", frame), ("late", 8)):
                order += 1
                source = f"{category}/{index}/{role}"
                samples.append({"order": order, "source": source, "category": category, "role": role})
                predictions.append({"source": source, "predicted_first_trace_frame_0based":
                                    8 if category == "mid_high" and role == "early" else 7})
                labels[f"{order - 1:03d}/{source}"] = {"displayed_frame_1based": labelled, "uncertain": False}
    return {"seed": 1, "samples": samples}, predictions, labels


def test_supported_not_supported_and_inconclusive():
    module = _module()
    assert module.score(*_case([9, 9, 10, 9, 8, 8], [8, 8, 9, 8, 8]))["verdict"] == "supported"
    assert module.score(*_case([8, 8, 9, 8, 8, 9], [8, 8, 8, 8, 8]))["verdict"] == "not_supported"
    assert module.score(*_case([9, 9, 9, 8, 8, 8], [8, 8, 8, 8, 8]))["verdict"] == "inconclusive"
    assert module.score(*_case([9, 9, 9, 9, 9, 9], [9, 9, 8, 8, 8]))["verdict"] == "inconclusive"


def test_exact_matches_count_against_sealed_predictions():
    result = _module().score(*_case([9, 9, 9, 9, 9, 9], [8, 8, 8, 8, 8]))
    assert result["exact_prediction_matches"] == [22, 22]
