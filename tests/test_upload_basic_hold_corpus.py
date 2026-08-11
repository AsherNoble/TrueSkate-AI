from pathlib import Path

import pytest

from scripts.cloud.upload_basic_hold_corpus import validate_corpus


def _write_sample(root: Path, name: str, *, point: tuple[float, float], duration: float) -> None:
    sample = root / name
    sample.mkdir(parents=True)
    (sample / "frame_000.png").write_bytes(b"placeholder")
    (sample / "meta.json").write_text(
        '{"gesture_distribution":"hold","spin_active":false,'
        f'"point":[{point[0]},{point[1]}],"hold_duration_s":{duration},'
        '"tap_calibration":{"accepted":true}}'
    )


def test_validate_corpus_requires_unique_commands(tmp_path: Path) -> None:
    _write_sample(tmp_path, "a", point=(0.3, 0.4), duration=0.5)
    _write_sample(tmp_path, "b", point=(0.3, 0.4), duration=0.5)

    with pytest.raises(ValueError, match="distinct commands"):
        validate_corpus(tmp_path, min_samples=2, require_unique_commands=True)

    assert validate_corpus(tmp_path, min_samples=2, require_unique_commands=False)["accepted"] == 2
