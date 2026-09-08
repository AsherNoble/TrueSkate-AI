from pathlib import Path

import numpy as np
import pytest

from trueskate_ai.vision.gameplay_filter import (
    bolt_modal_score, is_bolt_modal_frame, is_menu_frame)


REPO = Path(__file__).resolve().parents[1]


def _existing(paths: list[str]) -> list[Path]:
    found = [REPO / path for path in paths if (REPO / path).exists()]
    if not found:
        pytest.skip("local visual-regression fixtures are not present")
    return found


def test_known_legacy_gray_hub_frames_are_menu() -> None:
    frames = _existing([
        "data/self_labeled_traces/iPhone_XR_20260614_152013_sls_supercrown/"
        "sample_00007/frame_003.png",
        "data/self_labeled_traces/iPhone_XR_20260614_152013_sls_supercrown/"
        "sample_00008/frame_007.png",
        "data/sls_traces/iPhone_11_20260616_220810/sls_2016_super_crown/"
        "sample_00006/frame_006.png",
    ])
    for frame in frames:
        assert is_menu_frame(frame), frame


def test_known_clean_gameplay_frames_do_not_match_hub_nav() -> None:
    frames = _existing([
        "data/self_labeled_traces/iPhone_XR_20260614_152013_sls_supercrown/"
        "sample_00006/frame_003.png",
        "tmp/editor_detector/labeled/gameplay/game_home_xr1.png",
        "tmp/editor_detector/labeled/gameplay/game_home_xr2.png",
        "tmp/editor_detector/heldout/gameplay/gm_iPhone_XR_20260714_030144_000020.png",
        "tmp/editor_detector/heldout/gameplay/gm_iPhone_XR2_20260714_025412_000010.png",
    ])
    for frame in frames:
        assert not is_menu_frame(frame), frame


def test_synthetic_repeated_hub_cells_are_detected() -> None:
    frame = np.full((200, 100, 3), 160, dtype=np.uint8)
    frame[180:, :] = 12
    for cell in range(5):
        x0 = cell * 20 + 5
        frame[184:194, x0:x0 + 10] = 170
    assert is_menu_frame(frame)


def test_synthetic_dark_gameplay_with_sparse_ui_is_not_hub() -> None:
    frame = np.full((200, 100, 3), 150, dtype=np.uint8)
    frame[180:, :] = 12
    # A home indicator and one right-side HUD element do not form five nav cells.
    frame[195:198, 40:60] = 190
    frame[182:192, 84:96] = 170
    assert not is_menu_frame(frame)


def test_synthetic_bolt_center_modal_detected_and_missed_by_menu() -> None:
    # Light dialog panel filling the central band (Bolt Challenges modal shape).
    frame = np.full((200, 100, 3), 150, dtype=np.uint8)  # mid-tone gameplay
    frame[int(0.34 * 200):int(0.73 * 200), int(0.20 * 100):int(0.80 * 100)] = 235
    assert is_bolt_modal_frame(frame)
    assert bolt_modal_score(frame) > 0.5
    # The center dialog is invisible to the bottom-bar menu detector (complementary).
    assert not is_menu_frame(frame)


def test_synthetic_clean_gameplay_with_bright_sky_is_not_bolt_modal() -> None:
    # A bright region OUTSIDE the central band must not trip the modal detector.
    frame = np.full((200, 100, 3), 150, dtype=np.uint8)
    frame[:int(0.20 * 200), :] = 240  # bright sky at the very top
    assert not is_bolt_modal_frame(frame)
    assert bolt_modal_score(frame) < 0.05
