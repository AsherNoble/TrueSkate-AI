"""Bounded tar shards which stage corpus samples off Modal Volume/FUSE.

The archive preserves the original compact H.264/PNG sample files.  A worker
reads a few large sequential objects from the volume, safely materialises them
on ephemeral disk, then uses the ordinary dataset.  Therefore sharding changes
storage access, not preprocessing, tensors, or targets.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from trueskate_ai.data.cohort_manifest import (
    manifest_entries, read_manifest, sample_content_fingerprint,
)

SHARD_SCHEMA_VERSION = 1
DEFAULT_MAX_SAMPLES = 512
DEFAULT_MAX_BYTES = 2 * 1024**3


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _sample_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _shard_fingerprint(payload: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in payload.items()
                if key not in {"fingerprint", "created_at", "root_hint"}}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_shard_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("kind") != "model1_sequential_shards":
        raise ValueError("not a Model 1 sequential-shard manifest")
    if int(payload.get("schema_version", -1)) != SHARD_SCHEMA_VERSION:
        raise ValueError("unsupported sequential-shard schema")
    if not isinstance(payload.get("experiment_manifest"), str):
        raise ValueError("sequential shards must name their experiment manifest")
    if payload.get("fingerprint") != _shard_fingerprint(payload):
        raise ValueError("sequential-shard manifest fingerprint mismatch")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("sequential-shard manifest has no shards")
    seen: set[str] = set()
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise ValueError("invalid shard entry")
        samples = shard.get("samples")
        if not isinstance(samples, list) or int(shard.get("sample_count", -1)) != len(samples):
            raise ValueError("shard sample_count mismatch")
        if not isinstance(shard.get("sha256"), str) or not shard["sha256"].startswith("sha256:"):
            raise ValueError("shard has no SHA-256")
        overlap = seen.intersection(samples)
        if overlap:
            raise ValueError(f"samples occur in multiple shards: {sorted(overlap)[:3]}")
        seen.update(samples)
    if int(payload.get("sample_count", -1)) != len(seen):
        raise ValueError("top-level shard sample_count mismatch")


def read_shard_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read shard manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("shard manifest must be a JSON object")
    validate_shard_manifest(payload)
    return payload


def build_sequential_shards(
    corpus_root: str | Path,
    experiment_manifest: str | Path,
    out_dir: str | Path,
    *,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    if max_samples < 1 or max_bytes < 1:
        raise ValueError("max_samples and max_bytes must be positive")
    corpus_root = Path(corpus_root).resolve()
    experiment = read_manifest(experiment_manifest)
    if experiment.get("kind") != "model1_experiment":
        raise ValueError("shards require a model1_experiment manifest")
    all_entries = []
    for partition in ("train", "validation", "test"):
        all_entries.extend(manifest_entries(experiment, partition=partition))
    sample_paths = [(corpus_root / entry["path"]).resolve() for entry in all_entries]
    for sample in sample_paths:
        try:
            sample.relative_to(corpus_root)
        except ValueError as exc:
            raise ValueError(f"experiment sample escapes corpus root: {sample}") from exc
        if not sample.is_dir():
            raise FileNotFoundError(sample)
    sizes = [_sample_size(sample) for sample in sample_paths]
    too_large = [str(path) for path, size in zip(sample_paths, sizes) if size > max_bytes]
    if too_large:
        raise ValueError(f"individual samples exceed max_bytes: {too_large[:3]}")

    groups: list[list[tuple[Path, dict[str, Any], int]]] = []
    current: list[tuple[Path, dict[str, Any], int]] = []
    current_bytes = 0
    for sample, entry, size in zip(sample_paths, all_entries, sizes):
        if current and (len(current) >= max_samples or current_bytes + size > max_bytes):
            groups.append(current)
            current, current_bytes = [], 0
        current.append((sample, entry, size))
        current_bytes += size
    if current:
        groups.append(current)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_records = []
    for index, group in enumerate(groups):
        target = out_dir / f"shard_{index:05d}.tar"
        temporary = target.with_suffix(".tar.tmp")
        with tarfile.open(temporary, mode="w") as archive:
            for sample, entry, _size in group:
                archive.add(sample, arcname=entry["path"], recursive=True)
        temporary.replace(target)
        shard_records.append({
            "path": target.name,
            "sha256": _file_sha256(target),
            "bytes": target.stat().st_size,
            "sample_count": len(group),
            "samples": [entry["path"] for _sample, entry, _size in group],
        })
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "kind": "model1_sequential_shards",
        "experiment_fingerprint": experiment["fingerprint"],
        "experiment_manifest": "experiment.json",
        "sample_count": len(all_entries),
        "max_samples": max_samples,
        "max_bytes": max_bytes,
        "shards": shard_records,
    }
    payload["fingerprint"] = _shard_fingerprint(payload)
    experiment_copy = out_dir / payload["experiment_manifest"]
    experiment_tmp = experiment_copy.with_suffix(".json.tmp")
    experiment_tmp.write_text(json.dumps(experiment, indent=2, sort_keys=True) + "\n")
    experiment_tmp.replace(experiment_copy)
    manifest_path = out_dir / "shards.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return payload


def _safe_extract(archive: tarfile.TarFile, destination: Path, allowed_samples: set[str]) -> None:
    for member in archive:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"unsafe archive member {member.name!r}")
        if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
            raise ValueError(f"unsupported archive member {member.name!r}")
        if not any(path == PurePosixPath(sample) or PurePosixPath(sample) in path.parents
                   for sample in allowed_samples):
            raise ValueError(f"archive member is outside declared samples: {member.name!r}")
        target = destination.joinpath(*path.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        source = archive.extractfile(member)
        if source is None:
            raise ValueError(f"could not read archive member {member.name!r}")
        with source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=1024 * 1024)


def materialize_sequential_shards(
    shard_manifest: str | Path,
    destination: str | Path,
    *,
    verify_samples: bool = True,
) -> Path:
    """Safely stage all shards and verify they reproduce declared sample bytes."""
    shard_manifest = Path(shard_manifest)
    payload = read_shard_manifest(shard_manifest)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    declared_samples: set[str] = set()
    for shard in payload["shards"]:
        archive_path = shard_manifest.parent / shard["path"]
        actual = _file_sha256(archive_path)
        if actual != shard["sha256"]:
            raise ValueError(f"shard content changed: {archive_path}")
        samples = set(shard["samples"])
        with tarfile.open(archive_path, mode="r:") as archive:
            _safe_extract(archive, destination, samples)
        declared_samples.update(samples)
    if verify_samples:
        experiment_entries = {
            entry["path"]: entry for partition in ("train", "validation", "test")
            for entry in manifest_entries_from_fingerprint_source(shard_manifest, payload, partition)
        }
        for relative in declared_samples:
            expected = experiment_entries.get(relative)
            if expected is not None:
                actual = sample_content_fingerprint(destination / relative)
                if actual != expected["content_sha256"]:
                    raise ValueError(f"materialized sample differs from cohort manifest: {relative}")
    return destination


def manifest_entries_from_fingerprint_source(
    shard_manifest: Path,
    shard_payload: Mapping[str, Any],
    partition: str,
) -> list[dict[str, Any]]:
    """Read the sibling experiment manifest named by ``experiment_manifest``."""
    relative = shard_payload.get("experiment_manifest")
    if not isinstance(relative, str):
        # Older/externally transported shard sets can still validate archive
        # hashes. Their sample hashes are checked later by BasicLinearClipDataset.
        return []
    experiment_path = shard_manifest.parent / relative
    experiment = read_manifest(experiment_path)
    if experiment["fingerprint"] != shard_payload["experiment_fingerprint"]:
        raise ValueError("shard and experiment manifest fingerprints disagree")
    return manifest_entries(experiment, partition=partition)
