from collections import Counter
import math

from scripts.inspect.probe_control_contamination import generate_boundary_sequence
from trueskate_ai.data.control_hitboxes import (
    EFFECTIVE_CONTROL_HITBOXES,
    start_is_safe,
)


def test_boundary_sequence_is_exact_balanced_safe_and_targets_controls():
    sequence = generate_boundary_sequence(300, 20260917)
    by_name = {item.name: item for item in EFFECTIVE_CONTROL_HITBOXES}
    counts = Counter(item.target_control for item in sequence)

    assert len(sequence) == 300
    assert set(counts) == set(by_name)
    assert max(counts.values()) - min(counts.values()) <= 1
    for index, item in enumerate(sequence):
        assert item.index == index
        assert start_is_safe(item.start)
        assert by_name[item.target_control].contains(item.end)
        assert math.dist(item.start, item.end) >= 0.05
        assert item.duration_s in {0.25, 0.35, 0.50}


def test_boundary_sequence_is_reproducible_and_seeded():
    first = generate_boundary_sequence(40, 7)
    assert first == generate_boundary_sequence(40, 7)
    assert first != generate_boundary_sequence(40, 8)
