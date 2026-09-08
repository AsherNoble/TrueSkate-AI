"""Frozen-cohort construction, scaling decisions, and Modal cost estimates.

This module is deliberately free of Modal calls.  It prepares auditable inputs
and estimates spend; launching a paid tranche remains an explicit owner action.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from trueskate_ai.data.cohort_manifest import (
    COHORT_ROLES,
    MANIFEST_SCHEMA_VERSION,
    assert_zero_cohort_leakage,
    manifest_entries,
    sample_content_fingerprint,
    seal_manifest,
    validate_manifest,
)
from trueskate_ai.vision.basic_linear_audit import command_key
from trueskate_ai.vision.basic_linear_dataset import discover_basic_linear_samples

DEFAULT_LINEAR_RUNGS = (13_100, 26_200, 52_400, 104_800, 209_600)
DEPLOYMENT_PARKS = (
    "SLS 2015 Super Crown",
    "SLS 2013 Kansas City",
    "SLS 2015 Los Angeles",
    "Skateboard GB 2024",
)
CHALLENGE_PARK = "SLS 2016 Munich"

# Modal public resource prices retrieved 2026-09-04.  CPU and memory are
# additive and the trainer explicitly requests 64 GiB memory while using the
# default 0.125 physical-core request.
MODAL_PRICE_SOURCE = "https://modal.com/pricing"
MODAL_RESOURCE_RATES_PER_SECOND = {
    "cpu_core": 0.0000131,
    "memory_gib": 0.00000222,
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}


def _capture_date(meta: Mapping[str, Any]) -> str | None:
    for key in ("capture_date", "collected_at", "segment_started_at", "started_at"):
        value = meta.get(key)
        if not isinstance(value, str) or not value:
            continue
        match = re.match(r"(20\d{2})[-:]?(\d{2})[-:]?(\d{2})", value)
        if match:
            return "-".join(match.groups())
    session = str(meta.get("session") or "")
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", session)
    return "-".join(match.groups()) if match else None


def _entry(root: Path, sample: Path, *, require_provenance: bool) -> dict[str, Any]:
    import json

    meta = json.loads((sample / "meta.json").read_text())
    device = meta.get("device")
    park = meta.get("park")
    session = meta.get("session")
    capture_date = _capture_date(meta)
    missing = [name for name, value in (
        ("device", device), ("park", park), ("session", session), ("date", capture_date)
    ) if not isinstance(value, str) or not value]
    if require_provenance and missing:
        raise ValueError(f"{sample}: strict cohort provenance missing {', '.join(missing)}")
    return {
        "path": sample.relative_to(root).as_posix(),
        "command_key": command_key(meta),
        "content_sha256": sample_content_fingerprint(sample),
        "device": str(device or "<missing>"),
        "park": str(park or "<missing>"),
        "session": str(session or "<missing>"),
        "date": capture_date or "<missing>",
        "gesture_kind": str(meta.get("gesture_distribution") or "<missing>"),
    }


def cohort_coverage(entries: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(Counter(str(entry[field]) for entry in entries).items()))
        for field in ("device", "park", "session", "date", "gesture_kind")
    }


def build_linear_cohort_manifest(
    root: str | Path,
    *,
    cohort: str,
    role: str,
    corpus_root: str | Path | None = None,
    require_provenance: bool = True,
    allowed_parks: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build one content-addressed manifest from strict linear admissions."""
    if role not in COHORT_ROLES:
        raise ValueError(f"role must be one of {sorted(COHORT_ROLES)}, got {role!r}")
    if not cohort.strip():
        raise ValueError("cohort name may not be empty")
    selection_root = Path(root).resolve()
    corpus_root = Path(corpus_root).resolve() if corpus_root is not None else selection_root
    try:
        selection_root.relative_to(corpus_root)
    except ValueError as exc:
        raise ValueError("cohort selection root must be inside corpus_root") from exc
    samples, strict_counts = discover_basic_linear_samples(selection_root)
    entries = [_entry(corpus_root, sample, require_provenance=require_provenance) for sample in samples]
    commands = [entry["command_key"] for entry in entries]
    if len(commands) != len(set(commands)):
        duplicates = len(commands) - len(set(commands))
        raise ValueError(f"cohort contains {duplicates} duplicate exact commands")
    permitted = set(allowed_parks or ())
    if permitted:
        unexpected = sorted({entry["park"] for entry in entries} - permitted)
        if unexpected:
            raise ValueError(f"cohort contains parks outside its contract: {unexpected}")
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "model1_cohort",
        "cohort": cohort,
        "role": role,
        "subtype": "linear",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root_hint": str(corpus_root),
        "sample_count": len(entries),
        "strict_counts": strict_counts,
        "coverage": cohort_coverage(entries),
        "samples": entries,
    }
    return seal_manifest(payload)


def _stable_rank(seed: int, stratum: tuple[str, ...], entry: Mapping[str, Any]) -> bytes:
    identity = "\0".join((str(seed), *stratum, str(entry["content_sha256"]), str(entry["path"])))
    return hashlib.sha256(identity.encode("utf-8")).digest()


def balanced_nested_order(
    entries: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    strata: Sequence[str] = ("device", "park"),
) -> list[dict[str, Any]]:
    """Return one stable round-robin order whose prefixes stay stratum-balanced."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for source in entries:
        entry = dict(source)
        key = tuple(str(entry.get(field, "<missing>")) for field in strata)
        groups[key].append(entry)
    queues: dict[tuple[str, ...], deque[dict[str, Any]]] = {}
    for key, values in groups.items():
        queues[key] = deque(sorted(values, key=lambda entry: _stable_rank(seed, key, entry)))
    order: list[dict[str, Any]] = []
    keys = sorted(queues)
    while any(queues.values()):
        for key in keys:
            if queues[key]:
                order.append(queues[key].popleft())
    return order


def build_nested_subset_manifests(
    training_cohort: Mapping[str, Any],
    sizes: Iterable[int] = DEFAULT_LINEAR_RUNGS,
    *,
    seed: int = 0,
) -> list[dict[str, Any]]:
    validate_manifest(training_cohort)
    if training_cohort.get("kind") != "model1_cohort" or training_cohort.get("role") != "training":
        raise ValueError("nested subsets require a training cohort manifest")
    entries = manifest_entries(training_cohort)
    requested = list(sizes)
    if not requested or any(isinstance(size, bool) or size <= 0 for size in requested):
        raise ValueError("subset sizes must be positive integers")
    if requested != sorted(set(requested)):
        raise ValueError("subset sizes must be strictly increasing")
    if requested[-1] > len(entries):
        raise ValueError(f"largest subset {requested[-1]} exceeds cohort size {len(entries)}")
    order = balanced_nested_order(entries, seed=seed)
    manifests = []
    for size in requested:
        selected = order[:size]
        manifests.append(seal_manifest({
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": "model1_subset",
            "cohort": str(training_cohort["cohort"]),
            "role": "training",
            "subtype": training_cohort.get("subtype", "linear"),
            "parent_fingerprint": training_cohort["fingerprint"],
            "nesting_seed": seed,
            "sample_count": size,
            "coverage": cohort_coverage(selected),
            "samples": selected,
        }))
    return manifests


def assert_deterministic_nesting(subsets: Sequence[Mapping[str, Any]]) -> None:
    previous: list[str] = []
    parent = None
    for payload in subsets:
        validate_manifest(payload)
        if payload.get("kind") != "model1_subset":
            raise ValueError("nesting checks accept only subset manifests")
        if parent is None:
            parent = payload.get("parent_fingerprint")
        elif payload.get("parent_fingerprint") != parent:
            raise ValueError("nested subsets have different parent cohorts")
        paths = [entry["path"] for entry in manifest_entries(payload)]
        if paths[:len(previous)] != previous:
            raise ValueError("subset is not a prefix-preserving deterministic nest")
        previous = paths


def build_experiment_manifest(
    training_subset: Mapping[str, Any],
    validation_cohort: Mapping[str, Any],
    *,
    certification_cohort: Mapping[str, Any] | None = None,
    name: str,
) -> dict[str, Any]:
    """Combine already-frozen cohorts into explicit trainer partitions."""
    cohorts = [training_subset, validation_cohort]
    if certification_cohort is not None:
        cohorts.append(certification_cohort)
    for payload in cohorts:
        validate_manifest(payload)
    if training_subset.get("role") != "training":
        raise ValueError("training partition must have role=training")
    if validation_cohort.get("role") != "validation":
        raise ValueError("validation partition must have role=validation")
    if certification_cohort is not None and certification_cohort.get("role") != "certification":
        raise ValueError("test partition must have role=certification")
    assert_zero_cohort_leakage(cohorts)
    partitions = {
        "train": manifest_entries(training_subset),
        "validation": manifest_entries(validation_cohort),
        "test": manifest_entries(certification_cohort) if certification_cohort else [],
    }
    return seal_manifest({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "model1_experiment",
        "name": name,
        "subtype": "linear",
        "source_fingerprints": {
            "train": training_subset["fingerprint"],
            "validation": validation_cohort["fingerprint"],
            "test": certification_cohort["fingerprint"] if certification_cohort else None,
        },
        "partitions": partitions,
    })


def relative_error_reductions(observations: Iterable[Mapping[str, Any]]) -> list[dict[str, float | int]]:
    """Aggregate seeds by N and report relative validation-error reductions."""
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in observations:
        size = int(row["training_samples"])
        recovery = float(row["late_validation_recovery"])
        if not 0.0 <= recovery <= 1.0:
            raise ValueError(f"invalid recovery {recovery}")
        grouped[size].append(1.0 - recovery)
    sizes = sorted(grouped)
    result = []
    previous_error = None
    for size in sizes:
        values = np.asarray(grouped[size], dtype=float)
        mean_error = float(values.mean())
        reduction = (None if previous_error is None or previous_error == 0.0 else
                     (previous_error - mean_error) / previous_error)
        result.append({
            "training_samples": size,
            "seeds": len(values),
            "mean_error": mean_error,
            "seed_error_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "relative_error_reduction": reduction,
        })
        previous_error = mean_error
    return result


def scaling_status(observations: Iterable[Mapping[str, Any]], *, plateau_fraction: float = 0.20) -> dict[str, Any]:
    rows = relative_error_reductions(observations)
    reductions = [row["relative_error_reduction"] for row in rows if row["relative_error_reduction"] is not None]
    plateau = len(reductions) >= 2 and all(value < plateau_fraction for value in reductions[-2:])
    return {
        "status": "plateau_diagnosis_required" if plateau else "continue_one_doubling",
        "plateau": plateau,
        "criterion": f"last two relative error reductions < {plateau_fraction:.0%}",
        "rungs": rows,
    }


def fit_error_scaling_law(
    observations: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 500,
    seed: int = 0,
) -> dict[str, Any]:
    """Fit ``e(N)=e_floor+a*N^-alpha`` with seed and binomial uncertainty.

    Each row supplies ``training_samples``, ``late_validation_recovery``, and
    optionally ``validation_samples``. Four distinct sizes are the minimum: a
    three-parameter curve through three points has no residual degree of freedom
    and cannot establish that the law describes the observations.
    """
    from scipy.optimize import curve_fit

    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    grouped: dict[int, list[tuple[float, int | None]]] = defaultdict(list)
    for row in observations:
        size = int(row["training_samples"])
        recovery = float(row["late_validation_recovery"])
        validation_samples = (int(row["validation_samples"])
                              if row.get("validation_samples") is not None else None)
        if size < 1 or not 0.0 <= recovery <= 1.0:
            raise ValueError("scaling observations contain invalid size/recovery")
        if validation_samples is not None and validation_samples < 1:
            raise ValueError("validation_samples must be positive")
        grouped[size].append((recovery, validation_samples))
    if len(grouped) < 4:
        raise ValueError("an identifiable three-parameter scaling fit needs at least four sizes")

    sizes = np.asarray(sorted(grouped), dtype=float)
    errors, sigmas = [], []
    for size in sizes.astype(int):
        recoveries = np.asarray([value for value, _n in grouped[size]], dtype=float)
        error = 1.0 - float(recoveries.mean())
        seed_sem = (float(recoveries.std(ddof=1)) / math.sqrt(len(recoveries))
                    if len(recoveries) > 1 else 0.0)
        sample_counts = [count for _value, count in grouped[size] if count is not None]
        binomial_sem = (math.sqrt(max(0.0, (1.0 - error) * error) / min(sample_counts))
                        if sample_counts else 0.0)
        errors.append(error)
        sigmas.append(max(math.sqrt(seed_sem**2 + binomial_sem**2), 1e-6))
    errors_array = np.asarray(errors)
    sigmas_array = np.asarray(sigmas)

    def law(n, error_floor, amplitude, alpha):
        return error_floor + amplitude * np.power(n, -alpha)

    floor_guess = max(0.0, float(errors_array.min()) * .5)
    alpha_guess = .5
    amplitude_guess = max(1e-9, (errors_array[0] - floor_guess) * sizes[0]**alpha_guess)
    params, covariance = curve_fit(
        law, sizes, errors_array,
        p0=(floor_guess, amplitude_guess, alpha_guess),
        sigma=sigmas_array, absolute_sigma=True, maxfev=100_000,
        bounds=((0.0, 0.0, .001), (float(errors_array.min()), np.inf, 5.0)),
    )
    predicted = law(sizes, *params)
    residuals = errors_array - predicted
    dof = len(sizes) - 3
    result: dict[str, Any] = {
        "model": "e(N) = e_floor + a*N^-alpha",
        "distinct_sizes": len(sizes),
        "degrees_of_freedom": dof,
        "parameters": {
            "e_floor": float(params[0]), "a": float(params[1]), "alpha": float(params[2]),
        },
        "parameter_standard_errors": {
            name: float(value) for name, value in zip(
                ("e_floor", "a", "alpha"), np.sqrt(np.diag(covariance)),
            )
        },
        "weighted_chi_square": float(np.sum((residuals / sigmas_array) ** 2)),
        "reduced_chi_square": float(np.sum((residuals / sigmas_array) ** 2) / dof),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "rungs": [
            {"training_samples": int(n), "mean_error": float(error),
             "combined_standard_error": float(sigma), "fitted_error": float(fitted)}
            for n, error, sigma, fitted in zip(sizes, errors_array, sigmas_array, predicted)
        ],
    }

    rng = np.random.default_rng(seed)
    bootstrap_params = []
    for _ in range(bootstrap_samples):
        sampled_errors = []
        for size in sizes.astype(int):
            rows = grouped[size]
            chosen = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
            recoveries = []
            for recovery, count in chosen:
                recoveries.append(
                    rng.binomial(count, recovery) / count if count is not None else recovery
                )
            sampled_errors.append(1.0 - float(np.mean(recoveries)))
        try:
            fitted, _ = curve_fit(
                law, sizes, np.asarray(sampled_errors), p0=params,
                maxfev=20_000,
                bounds=((0.0, 0.0, .001), (max(sampled_errors), np.inf, 5.0)),
            )
            bootstrap_params.append(fitted)
        except (RuntimeError, ValueError):
            continue
    if bootstrap_params:
        samples = np.asarray(bootstrap_params)
        result["bootstrap_successes"] = len(samples)
        result["bootstrap_95_interval"] = {
            name: [float(low), float(high)]
            for name, low, high in zip(
                ("e_floor", "a", "alpha"),
                np.quantile(samples, .025, axis=0), np.quantile(samples, .975, axis=0),
            )
        }
    else:
        result["bootstrap_successes"] = 0
        result["bootstrap_95_interval"] = None
    return result


def gradient_clipping_decision(
    control_late_recovery: Iterable[float],
    clipped_late_recovery: Iterable[float],
    *,
    required_variability_reduction: float = 0.30,
    allowed_mean_drop: float = 0.01,
) -> dict[str, Any]:
    """Apply the predeclared clipping gate to late-epoch recovery values."""
    control = np.asarray(list(control_late_recovery), dtype=float)
    clipped = np.asarray(list(clipped_late_recovery), dtype=float)
    if len(control) < 2 or len(clipped) < 2:
        raise ValueError("clipping decision needs at least two values in each arm")
    if not (np.isfinite(control).all() and np.isfinite(clipped).all()
            and np.all((control >= 0) & (control <= 1))
            and np.all((clipped >= 0) & (clipped <= 1))):
        raise ValueError("recovery values must be finite probabilities")
    control_sd = float(control.std(ddof=1))
    clipped_sd = float(clipped.std(ddof=1))
    variability_reduction = (0.0 if control_sd == 0.0 else
                             (control_sd - clipped_sd) / control_sd)
    mean_change = float(clipped.mean() - control.mean())
    selected = (
        variability_reduction >= required_variability_reduction
        and mean_change >= -allowed_mean_drop
    )
    return {
        "selected": selected,
        "control_mean": float(control.mean()),
        "clipped_mean": float(clipped.mean()),
        "control_sd": control_sd,
        "clipped_sd": clipped_sd,
        "variability_reduction": variability_reduction,
        "mean_change": mean_change,
        "gate": {
            "minimum_variability_reduction": required_variability_reduction,
            "maximum_mean_recovery_drop": allowed_mean_drop,
        },
    }


def modal_hourly_rate(
    gpu: str,
    *,
    memory_gib: float = 64.0,
    cpu_cores: float = 0.125,
) -> float:
    if gpu not in MODAL_RESOURCE_RATES_PER_SECOND:
        raise ValueError(f"unknown Modal GPU {gpu!r}")
    if gpu in {"cpu_core", "memory_gib"} or memory_gib < 0 or cpu_cores < 0:
        raise ValueError("invalid Modal resource request")
    per_second = (
        MODAL_RESOURCE_RATES_PER_SECOND[gpu]
        + memory_gib * MODAL_RESOURCE_RATES_PER_SECOND["memory_gib"]
        + cpu_cores * MODAL_RESOURCE_RATES_PER_SECOND["cpu_core"]
    )
    return per_second * 3600.0


def estimate_modal_rungs(
    sizes: Iterable[int] = DEFAULT_LINEAR_RUNGS,
    *,
    base_size: int = 13_100,
    base_run_hours: float = 8.42,
    seeds: int = 3,
    gpu: str = "L4",
    memory_gib: float = 64.0,
    cpu_cores: float = 0.125,
) -> dict[str, Any]:
    """Estimate three-seed rung cost assuming runtime is linear in samples."""
    sizes = list(sizes)
    if base_size < 1 or base_run_hours <= 0 or seeds < 1 or any(size < 1 for size in sizes):
        raise ValueError("sizes, hours, and seeds must be positive")
    hourly = modal_hourly_rate(gpu, memory_gib=memory_gib, cpu_cores=cpu_cores)
    gpu_hourly = MODAL_RESOURCE_RATES_PER_SECOND[gpu] * 3600.0
    rungs = []
    cumulative = 0.0
    cumulative_gpu = 0.0
    for size in sizes:
        per_seed_hours = base_run_hours * size / base_size
        cost = per_seed_hours * seeds * hourly
        gpu_cost = per_seed_hours * seeds * gpu_hourly
        cumulative += cost
        cumulative_gpu += gpu_cost
        rungs.append({
            "training_samples": size,
            "per_seed_hours": per_seed_hours,
            "three_seed_hours": per_seed_hours * seeds,
            "estimated_cost_usd": cost,
            "gpu_only_cost_usd": gpu_cost,
            "cumulative_cost_usd": cumulative,
            "cumulative_gpu_only_cost_usd": cumulative_gpu,
        })
    return {
        "gpu": gpu,
        "seeds": seeds,
        "memory_gib": memory_gib,
        "cpu_cores": cpu_cores,
        "hourly_rate_usd": hourly,
        "gpu_only_hourly_rate_usd": gpu_hourly,
        "base_measurement": {"training_samples": base_size, "per_seed_hours": base_run_hours},
        "assumption": "wall time scales linearly with admitted training samples",
        "price_source": MODAL_PRICE_SOURCE,
        "rungs": rungs,
        "total_cost_usd": cumulative,
        "total_gpu_only_cost_usd": cumulative_gpu,
    }
