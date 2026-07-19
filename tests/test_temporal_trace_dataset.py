import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from trueskate_ai.vision.temporal_trace_dataset import (
    TemporalTraceSequenceDataset,
    discover_sample_paths,
    split_by_sample,
)


def _write_sample(
    parent: Path,
    name: str,
    meta: dict,
    *,
    images: list[np.ndarray] | None = None,
) -> Path:
    sample = parent / name
    sample.mkdir(parents=True)
    (sample / "meta.json").write_text(json.dumps(meta))
    if images is None:
        images = [
            np.full((32, 16, 3), index * 20, np.uint8)
            for index in range(len(meta["frame_times"]))
        ]
    for index, image in enumerate(images):
        assert cv2.imwrite(str(sample / f"frame_{index:03d}.png"), image)
    return sample


def _flick_meta(frame_times, **updates):
    meta = {
        "gesture_distribution": "flick",
        "waypoints": [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]],
        "duration": 1.0,
        "easing_power": 1.0,
        "frame_times": list(frame_times),
        "gesture_video_time_s": 10.0,
    }
    meta.update(updates)
    return meta


def _slot(x0, x1, duration=1.0):
    return [x0, 0.3, (x0 + x1) / 2.0, 0.5, x1, 0.7, duration, 1.0]


def test_end_relative_flick_is_one_padded_causal_sequence(tmp_path):
    sample = _write_sample(
        tmp_path,
        "sample_000",
        _flick_meta([-1.0, -0.5, 0.0, 0.2]),
    )
    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=6,
        image_height=32,
        image_width=16,
        max_touches=2,
        latency_s=0.0,
        require_trace=False,
    )

    assert dataset.sample_paths == (sample,)
    assert dataset.sample_frame_times(0).tolist() == pytest.approx([0.0, 0.5, 1.0, 1.2])
    item = dataset[0]
    assert set(item) == {
        "frames", "heatmaps", "active", "centers", "touch_count",
        "delta_times", "valid_mask", "label_mask", "reset_mask",
    }
    assert item["frames"].shape == (6, 3, 32, 16)
    assert item["heatmaps"].shape == (6, 1, 32, 16)
    assert item["centers"].shape == (6, 2, 2)
    assert item["touch_count"].tolist() == [1, 1, 1, 0, 0, 0]
    assert item["valid_mask"].tolist() == [True, True, True, True, False, False]
    assert item["label_mask"].tolist() == [True, True, True, True, False, False]
    assert item["reset_mask"].tolist() == [True, False, False, False, False, False]
    assert item["delta_times"].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.2, 0.0, 0.0])
    assert item["centers"][0, 0].tolist() == pytest.approx([0.2, 0.3])
    assert item["centers"][1, 0].tolist() == pytest.approx([0.5, 0.5])
    assert item["centers"][2, 0].tolist() == pytest.approx([0.8, 0.7])
    assert torch.all(item["centers"][4:] == -1)
    assert float(item["heatmaps"][1].max()) > 0.98
    assert torch.count_nonzero(item["heatmaps"][3:]) == 0
    assert dataset.positive_frame_counts == [3]
    assert dataset.negative_frame_counts == [1]
    assert dataset.multi_touch_frame_counts == [0]
    assert dataset.stats["multi_touch_frames"] == 0
    assert dataset.stats["multi_touch_sequences"] == 0


def test_negative_delay_overlap_and_params_spin_form_three_stable_tracks(tmp_path):
    # slot 1 starts 0.5s into slot 0 (delay=-0.5).  Spin is held during the
    # overlap.  The W3C payload ends at t=1.1, hence the end-relative times.
    params = _slot(0.2, 0.4, 1.0) + _slot(0.8, 0.6, 0.6) + [-0.5] + [1.0, 0.4, 0.9]
    _write_sample(
        tmp_path,
        "sample_overlap",
        {
            "gesture_distribution": "spin",
            "params": params,
            "num_gestures": 2,
            "use_spin": True,
            "spin_active": True,
            "spin_button_xy": [0.06, 0.4],
            "frame_times": [-1.1, -0.6, -0.3, 0.0],
            "gesture_end_monotonic": 123.0,
        },
    )
    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=4,
        image_height=32,
        image_width=16,
        max_touches=3,
        latency_s=0.0,
        finger_stagger_s=0.0,
        require_trace=False,
    )
    item = dataset[0]

    assert item["touch_count"].tolist() == [1, 3, 3, 1]
    # Each physical interval keeps its assigned column for its full lifetime.
    assert item["centers"][1, 0].tolist() == pytest.approx([0.3, 0.5])
    assert item["centers"][1, 1].tolist() == pytest.approx([0.06, 0.4])
    assert item["centers"][1, 2].tolist() == pytest.approx([0.8, 0.3])
    assert item["centers"][2, 1].tolist() == pytest.approx([0.06, 0.4])
    assert item["centers"][2, 2].tolist() == pytest.approx([0.7, 0.5])
    assert item["centers"][3, 2].tolist() == pytest.approx([0.6, 0.7])
    assert torch.count_nonzero(item["heatmaps"][1] > 0.95) >= 3
    assert dataset.stats["kinds"] == {"spin": 1}
    assert dataset.stats["max_touch_count"] == 3
    assert dataset.multi_touch_frame_counts == [2]
    assert dataset.stats["multi_touch_frames"] == 2
    assert dataset.stats["multi_touch_sequences"] == 1


def test_overlap_fails_with_sample_identity_and_required_max_touches(tmp_path):
    params = _slot(0.2, 0.4) + _slot(0.8, 0.6, 0.6) + [-0.5] + [1.0, 0.4, 0.9]
    sample = _write_sample(
        tmp_path,
        "sample_too_many",
        {
            "gesture_distribution": "spin",
            "params": params,
            "num_gestures": 2,
            "use_spin": True,
            "spin_active": True,
            "frame_times": [-0.6],
            "gesture_end_monotonic": 123.0,
        },
    )
    with pytest.raises(ValueError, match=rf"{sample} requires max_touches=3"):
        TemporalTraceSequenceDataset(
            tmp_path,
            sequence_length=2,
            max_touches=2,
            latency_s=0.0,
            finger_stagger_s=0.0,
            require_trace=False,
        )


def test_spin_flick_payload_end_anchor_keeps_drag_and_spin_hold_distinct(tmp_path):
    _write_sample(
        tmp_path,
        "sample_spin_flick",
        _flick_meta(
            [-1.5, -1.0, -0.3, 0.1],
            gesture_distribution="spin_flick",
            spin_active=True,
            spin_hold_start_s=0.2,
            spin_hold_end_s=1.5,
            payload_total_s=1.5,
            spin_button_xy=[0.06, 0.4],
        ),
    )
    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=4,
        max_touches=2,
        latency_s=0.0,
        require_trace=False,
    )
    item = dataset[0]

    assert dataset.sample_frame_times(0).tolist() == pytest.approx([0.0, 0.5, 1.2, 1.6])
    assert item["touch_count"].tolist() == [1, 2, 1, 0]
    assert item["centers"][1, 1].tolist() == pytest.approx([0.06, 0.4])
    assert item["centers"][2, 0].tolist() == [-1.0, -1.0]
    assert item["centers"][2, 1].tolist() == pytest.approx([0.06, 0.4])


def test_trace_gate_masks_unreliable_frame_without_dropping_or_reordering(tmp_path):
    black = np.zeros((40, 20, 3), np.uint8)
    orange = black.copy()
    orange[17:24, 7:14] = (0, 140, 255)
    _write_sample(
        tmp_path,
        "sample_gate",
        _flick_meta([-1.0, -0.5, 0.2]),
        images=[black, orange, black],
    )
    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=5,
        image_height=40,
        image_width=20,
        latency_s=0.0,
        require_trace=True,
        trace_warm_threshold=1,
        trace_radius_px=5,
    )
    item = dataset[0]

    assert item["valid_mask"].tolist() == [True, True, True, False, False]
    assert item["touch_count"].tolist() == [1, 1, 0, 0, 0]
    assert item["label_mask"].tolist() == [False, True, True, False, False]
    assert item["delta_times"].tolist() == pytest.approx([0.0, 0.5, 0.7, 0.0, 0.0])
    assert dataset.stats["gated_frames"] == 1
    assert dataset.positive_frame_counts == [1]
    assert dataset.negative_frame_counts == [1]


def test_discovery_filters_park_and_flags_and_split_stays_sample_level(tmp_path):
    crown = tmp_path / "device" / "session" / "sls_2016_super_crown"
    munich = tmp_path / "device" / "session" / "sls_2016_munich"
    good_paths = []
    for index in range(4):
        good_paths.append(
            _write_sample(crown, f"sample_{index:03d}", _flick_meta([-1.0, 0.2]))
        )
    menu = _write_sample(crown, "sample_100", _flick_meta([-1.0]))
    (menu / ".menu").touch()
    editor = _write_sample(crown, "sample_101", _flick_meta([-1.0]))
    (editor / ".editor").touch()
    _write_sample(munich, "sample_999", _flick_meta([-1.0]))

    discovered = discover_sample_paths(tmp_path, include_path_term="SLS Super Crown")
    assert len(discovered) == 6
    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=3,
        include_path_term="SLS Super Crown",
        latency_s=0.0,
        require_trace=False,
    )
    assert dataset.sample_paths == tuple(good_paths)
    assert dataset.stats["menu_skipped"] == 1
    assert dataset.stats["editor_skipped"] == 1

    train_a, val_a = split_by_sample(dataset, val_fraction=0.25, seed=17)
    train_b, val_b = split_by_sample(dataset, val_fraction=0.25, seed=17)
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    assert set(train_a.indices).isdisjoint(val_a.indices)
    assert sorted(train_a.indices + val_a.indices) == list(range(len(dataset)))


def test_split_stratifies_genuine_labeled_overlap_sequences(tmp_path):
    params = _slot(0.2, 0.4, 1.0) + _slot(0.8, 0.6, 0.6) + [-0.5]
    for index in range(4):
        _write_sample(
            tmp_path,
            f"sample_overlap_{index:03d}",
            {
                "gesture_distribution": "nslot",
                "params": params,
                "num_gestures": 2,
                "use_spin": False,
                "frame_times": [0.0, 0.6],
                "gesture_start_monotonic": 123.0,
            },
        )
    for index in range(6):
        _write_sample(
            tmp_path,
            f"sample_single_{index:03d}",
            _flick_meta([-1.0, -0.4]),
        )

    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=2,
        max_touches=2,
        latency_s=0.0,
        finger_stagger_s=0.0,
        require_trace=False,
    )
    train_a, val_a = split_by_sample(dataset, val_fraction=0.4, seed=23)
    train_b, val_b = split_by_sample(dataset, val_fraction=0.4, seed=23)

    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    assert len(train_a) == 6
    assert len(val_a) == 4
    for indices in (train_a.indices, val_a.indices):
        overlap_flags = [dataset.multi_touch_frame_counts[index] > 0 for index in indices]
        assert any(overlap_flags)
        assert not all(overlap_flags)


def test_corrupt_or_non_mapping_metadata_is_skipped_and_counted(tmp_path):
    good = _write_sample(tmp_path, "sample_good", _flick_meta([-1.0, 0.2]))
    corrupt = tmp_path / "sample_corrupt"
    corrupt.mkdir()
    (corrupt / "meta.json").write_text("{interrupted")
    non_mapping = tmp_path / "sample_list"
    non_mapping.mkdir()
    (non_mapping / "meta.json").write_text("[]")

    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=3,
        latency_s=0.0,
        require_trace=False,
    )

    assert dataset.sample_paths == (good,)
    assert dataset.stats["bad_meta_skipped"] == 2


def test_malformed_supported_gesture_metadata_still_fails_loudly(tmp_path):
    sample = _write_sample(
        tmp_path,
        "sample_bad_duration",
        _flick_meta([-1.0], duration=-0.1),
    )

    with pytest.raises(ValueError, match=rf"{sample}: flick duration"):
        TemporalTraceSequenceDataset(
            tmp_path,
            sequence_length=2,
            latency_s=0.0,
            require_trace=False,
        )


def test_bounded_discovery_spreads_deterministically_across_matching_sessions(tmp_path):
    for session_index in range(4):
        park = tmp_path / f"device_session_{session_index}" / "sls_2016_super_crown"
        for sample_index in range(5):
            _write_sample(
                park,
                f"sample_{sample_index:03d}",
                _flick_meta([-1.0]),
            )

    first = discover_sample_paths(
        tmp_path, include_path_term="super crown", max_samples=4
    )
    second = discover_sample_paths(
        tmp_path, include_path_term="super crown", max_samples=4
    )
    assert first == second
    assert len(first) == 4
    # sqrt(4)=2 matching park roots are selected, then sampled round-robin;
    # one lexical session cannot consume the whole budget.
    assert len({path.parents[1].name for path in first}) == 2


def test_digit_only_path_filter_does_not_match_every_session(tmp_path):
    wanted = tmp_path / "device" / "session_202607170723" / "sls_2016_super_crown"
    other = tmp_path / "device" / "session_202607180915" / "sls_2016_super_crown"
    wanted_sample = _write_sample(wanted, "sample_000", _flick_meta([-1.0]))
    _write_sample(other, "sample_001", _flick_meta([-1.0]))

    assert discover_sample_paths(
        tmp_path, include_path_term="202607170723"
    ) == [wanted_sample]


def test_uint8_cache_avoids_epoch_rereads_and_reports_bytes(tmp_path, monkeypatch):
    _write_sample(
        tmp_path,
        "sample_cached",
        _flick_meta([-1.0, 0.2]),
    )
    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=3,
        image_height=32,
        image_width=16,
        latency_s=0.0,
        require_trace=False,
        cache_frames=True,
    )
    assert dataset.stats["cached_frame_bytes"] == 2 * 32 * 16 * 3
    assert dataset.stats["cached_frame_mib"] == pytest.approx(
        dataset.stats["cached_frame_bytes"] / (1024**2)
    )

    def no_epoch_read(*_args, **_kwargs):
        raise AssertionError("cached __getitem__ must not reread Modal FUSE")

    monkeypatch.setattr(cv2, "imread", no_epoch_read)
    item = dataset[0]
    assert item["frames"].shape == (3, 3, 32, 16)
    assert item["valid_mask"].tolist() == [True, True, False]


def test_parallel_cache_matches_serial_and_overlaps_frame_reads(tmp_path, monkeypatch):
    for index in range(5):
        _write_sample(
            tmp_path,
            f"sample_{index:03d}",
            _flick_meta([-1.0, -0.5, 0.2]),
        )
    dataset_kwargs = dict(
        sequence_length=4,
        image_height=32,
        image_width=16,
        latency_s=0.0,
        require_trace=True,
        trace_warm_threshold=1,
        cache_frames=True,
    )
    serial = TemporalTraceSequenceDataset(tmp_path, **dataset_kwargs)

    real_imread = cv2.imread
    lock = threading.Lock()
    active_reads = 0
    max_active_reads = 0

    def delayed_imread(*args, **kwargs):
        nonlocal active_reads, max_active_reads
        with lock:
            active_reads += 1
            max_active_reads = max(max_active_reads, active_reads)
        try:
            time.sleep(0.01)
            return real_imread(*args, **kwargs)
        finally:
            with lock:
                active_reads -= 1

    monkeypatch.setattr(cv2, "imread", delayed_imread)
    parallel = TemporalTraceSequenceDataset(
        tmp_path, cache_workers=3, **dataset_kwargs
    )

    assert max_active_reads >= 2
    assert parallel.sample_paths == serial.sample_paths
    assert parallel.stats == serial.stats
    assert parallel.positive_frame_counts == serial.positive_frame_counts
    assert parallel.negative_frame_counts == serial.negative_frame_counts
    for index in range(len(serial)):
        serial_item = serial[index]
        parallel_item = parallel[index]
        assert serial_item.keys() == parallel_item.keys()
        for key in serial_item:
            assert torch.equal(parallel_item[key], serial_item[key]), key


def test_parallel_cache_raises_first_candidate_error_deterministically(tmp_path):
    first = _write_sample(
        tmp_path,
        "sample_000",
        _flick_meta([-1.0, -1.1]),
    )
    _write_sample(
        tmp_path,
        "sample_001",
        _flick_meta([-1.0, -1.2]),
    )

    with pytest.raises(ValueError, match=rf"{first}: frame_times must be chronological"):
        TemporalTraceSequenceDataset(
            tmp_path,
            sequence_length=3,
            latency_s=0.0,
            require_trace=False,
            cache_workers=2,
        )


def test_negative_cache_worker_count_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="cache_workers must be >= 0"):
        TemporalTraceSequenceDataset(tmp_path, cache_workers=-1)


def test_optional_menu_detection_checks_priority_frames_before_retaining(tmp_path, monkeypatch):
    _write_sample(tmp_path, "sample_good", _flick_meta([-1.0, 0.2]))
    _write_sample(
        tmp_path,
        "sample_hidden_menu",
        _flick_meta([-1.0, 0.2]),
        images=[np.full((32, 16, 3), 255, np.uint8)] * 2,
    )

    from trueskate_ai.vision import gameplay_filter

    monkeypatch.setattr(gameplay_filter, "is_editor_frame", lambda _image: False)
    monkeypatch.setattr(
        gameplay_filter,
        "is_menu_frame",
        lambda image: float(np.asarray(image).mean()) > 250.0,
    )
    dataset = TemporalTraceSequenceDataset(
        tmp_path,
        sequence_length=3,
        latency_s=0.0,
        require_trace=False,
        detect_menu_frames=True,
    )
    assert [path.name for path in dataset.sample_paths] == ["sample_good"]
    assert dataset.stats["detected_menu_skipped"] == 1


def test_sequence_length_is_never_silently_truncated(tmp_path):
    sample = _write_sample(
        tmp_path,
        "sample_long",
        _flick_meta([-1.0, -0.8, -0.6, -0.4]),
    )
    with pytest.raises(ValueError, match=rf"{sample} contains 4 frames.*sequence_length=3"):
        TemporalTraceSequenceDataset(
            tmp_path,
            sequence_length=3,
            latency_s=0.0,
            require_trace=False,
        )
