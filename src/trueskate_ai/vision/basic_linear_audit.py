"""Offline audit helpers for the strict calibrated linear-drag corpus.

The audit deliberately reuses :func:`discover_basic_linear_samples` as its
admission authority.  It therefore cannot accidentally count a clip that the
Model 1 loader would reject.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from trueskate_ai.data.gesture_sampling import (
    BASIC_LINEAR_MAX_ABS_SLOPE,
    BASIC_LINEAR_MAX_S,
    BASIC_LINEAR_MIN_S,
)
from trueskate_ai.vision.basic_linear_dataset import discover_basic_linear_samples


def command_key(meta: dict[str, Any]) -> str:
    """Return the training split's exact, stable command identity."""
    return ":".join(f"{float(v):.9f}" for point in meta["waypoints"] for v in point) + \
        f":{float(meta['duration']):.9f}"


def _histogram(values: list[float], edges: np.ndarray) -> list[dict[str, float | int]]:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    return [{"lower": float(edges[index]), "upper": float(edges[index + 1]),
             "count": int(counts[index])}
            for index in range(len(counts))]


def _position_grid(points: list[tuple[float, float]], *, bins: int,
                   sparse_cell_max_count: int) -> dict[str, Any]:
    counts = np.zeros((bins, bins), dtype=int)
    for x, y in points:
        x_index = min(bins - 1, max(0, int(x * bins)))
        y_index = min(bins - 1, max(0, int(y * bins)))
        counts[y_index, x_index] += 1
    cells = [{"x_index": x, "y_index": y, "x_lower": x / bins,
              "x_upper": (x + 1) / bins, "y_lower": y / bins,
              "y_upper": (y + 1) / bins, "count": int(counts[y, x])}
             for y in range(bins) for x in range(bins)]
    return {"bins_per_axis": bins, "cells": cells,
            "sparse_cells": [cell for cell in cells if cell["count"] <= sparse_cell_max_count]}


def _nearest_spacing(vectors: np.ndarray) -> dict[str, float | int | None]:
    """Nearest distinct-command Euclidean spacing in normalized command space."""
    count = len(vectors)
    if count < 2:
        return {"unique_commands": count, "min": None, "p05": None,
                "median": None, "mean": None, "max": None}
    nearest = np.full(count, np.inf, dtype=float)
    # Blocked calculation keeps the command useful for a larger future tranche
    # without allocating an all-corpus O(n²) matrix.
    block = 512
    for start in range(0, count, block):
        stop = min(count, start + block)
        distances = np.sqrt(((vectors[start:stop, None, :] - vectors[None, :, :]) ** 2).sum(axis=2))
        for local, global_index in enumerate(range(start, stop)):
            distances[local, global_index] = np.inf
        nearest[start:stop] = distances.min(axis=1)
    return {"unique_commands": count, "min": float(nearest.min()),
            "p05": float(np.percentile(nearest, 5)), "median": float(np.median(nearest)),
            "mean": float(nearest.mean()), "max": float(nearest.max())}


def audit_basic_linear_corpus(root: str | Path, *, position_bins: int = 4,
                              numeric_bins: int = 8,
                              sparse_cell_max_count: int = 5) -> dict[str, Any]:
    """Return JSON-serialisable strict-admission, provenance, and coverage evidence."""
    if position_bins < 1 or numeric_bins < 1 or sparse_cell_max_count < 0:
        raise ValueError("bin counts must be positive and sparse_cell_max_count non-negative")
    root = Path(root)
    samples, strict_counts = discover_basic_linear_samples(root)
    entries: list[dict[str, Any]] = []
    commands: dict[str, list[dict[str, Any]]] = defaultdict(list)
    devices: Counter[str] = Counter()
    parks: Counter[str] = Counter()
    parks_by_device: dict[str, set[str]] = defaultdict(set)
    starts: list[tuple[float, float]] = []
    ends: list[tuple[float, float]] = []
    durations: list[float] = []
    slopes: list[float] = []
    displacements: list[float] = []

    for sample in samples:
        meta = json.loads((sample / "meta.json").read_text())
        device = str(meta.get("device") or "<missing>")
        park = str(meta.get("park") or "<missing>")
        (x0, y0), (x1, y1) = ((float(value) for value in point) for point in meta["waypoints"])
        duration = float(meta["duration"])
        key = command_key(meta)
        entry = {"path": str(sample.relative_to(root)), "device": device, "park": park}
        commands[key].append(entry)
        entries.append(entry)
        devices[device] += 1
        parks[park] += 1
        parks_by_device[device].add(park)
        starts.append((x0, y0))
        ends.append((x1, y1))
        durations.append(duration)
        slopes.append((y1 - y0) / (x1 - x0))
        displacements.append(float(np.hypot(x1 - x0, y1 - y0)))

    duplicate_groups = []
    for key, members in commands.items():
        if len(members) > 1:
            duplicate_groups.append({"command": key, "count": len(members), "samples": members,
                                     "cross_device": len({member["device"] for member in members}) > 1})
    duplicate_groups.sort(key=lambda item: (-int(item["count"]), str(item["command"])))
    unique_vectors = []
    for key in sorted(commands):
        values = np.fromstring(key, dtype=float, sep=":")
        values[-1] = (values[-1] - BASIC_LINEAR_MIN_S) / (BASIC_LINEAR_MAX_S - BASIC_LINEAR_MIN_S)
        unique_vectors.append(values)

    duration_edges = np.linspace(BASIC_LINEAR_MIN_S, BASIC_LINEAR_MAX_S, numeric_bins + 1)
    slope_edges = np.linspace(-BASIC_LINEAR_MAX_ABS_SLOPE, BASIC_LINEAR_MAX_ABS_SLOPE, numeric_bins + 1)
    displacement_edges = np.linspace(0.0, float(np.sqrt(2.0)), numeric_bins + 1)
    accepted = len(samples)
    return {
        "root": str(root),
        "strict_counts": strict_counts,
        "accepted": accepted,
        "provenance": {
            "per_device": dict(sorted(devices.items())), "per_park": dict(sorted(parks.items())),
            "parks_by_device": {device: sorted(names) for device, names in sorted(parks_by_device.items())},
            "explicit_device_provenanced": accepted - devices["<missing>"],
            "missing_device_provenance": devices["<missing>"],
        },
        "duplicates": {
            "duplicate_groups": duplicate_groups, "duplicate_samples": sum(int(group["count"]) - 1 for group in duplicate_groups),
            "cross_device_groups": sum(bool(group["cross_device"]) for group in duplicate_groups),
            "unique_commands": len(commands),
        },
        "coverage": {
            "start_position": _position_grid(starts, bins=position_bins,
                                              sparse_cell_max_count=sparse_cell_max_count),
            "end_position": _position_grid(ends, bins=position_bins,
                                            sparse_cell_max_count=sparse_cell_max_count),
            "duration": _histogram(durations, duration_edges),
            "slope": _histogram(slopes, slope_edges),
            "displacement": _histogram(displacements, displacement_edges),
            "nearest_command_spacing": _nearest_spacing(np.asarray(unique_vectors, dtype=float).reshape(len(unique_vectors), 5)),
        },
    }
