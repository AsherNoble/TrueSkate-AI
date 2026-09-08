"""Portable, content-addressed manifests for Model 1 corpus cohorts.

Directory discovery is useful while collecting, but it is not an experimental
split: adding one directory can silently move samples between train and
validation.  These helpers make an explicit manifest the immutable authority
for a cohort or nested training subset.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

MANIFEST_SCHEMA_VERSION = 1
COHORT_ROLES = frozenset({"training", "validation", "challenge", "certification"})
MANIFEST_KINDS = frozenset({"model1_cohort", "model1_subset", "model1_experiment"})


def sample_content_fingerprint(sample: str | Path) -> str:
    """Hash every regular sample file, including its relative name and length."""
    sample = Path(sample)
    digest = hashlib.sha256()
    files = sorted(path for path in sample.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"sample contains no files: {sample}")
    for path in files:
        if path.is_symlink():
            raise ValueError(f"sample content may not contain symlinks: {path}")
        relative = path.relative_to(sample).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fields which define corpus identity rather than where/when it was built."""
    return {
        key: value for key, value in payload.items()
        if key not in {"fingerprint", "created_at", "root_hint"}
    }


def manifest_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        _identity_payload(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["fingerprint"] = manifest_fingerprint(result)
    return result


def manifest_entries(payload: Mapping[str, Any], *, partition: str | None = None) -> list[dict[str, Any]]:
    kind = payload.get("kind")
    if kind not in MANIFEST_KINDS:
        raise ValueError(f"unsupported manifest kind {kind!r}")
    if int(payload.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported manifest schema {payload.get('schema_version')!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
    if kind == "model1_experiment":
        if partition is None:
            raise ValueError("experiment manifests require a partition name")
        partitions = payload.get("partitions")
        if not isinstance(partitions, dict) or partition not in partitions:
            raise ValueError(f"experiment manifest has no {partition!r} partition")
        entries = partitions[partition]
    else:
        if partition is not None:
            raise ValueError(f"{kind} does not contain experiment partitions")
        entries = payload.get("samples")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError("manifest samples must be a list of objects")
    expected = payload.get("sample_count") if kind != "model1_experiment" else None
    if expected is not None and int(expected) != len(entries):
        raise ValueError(f"manifest sample_count={expected} but contains {len(entries)} entries")
    return [dict(entry) for entry in entries]


def validate_manifest(payload: Mapping[str, Any]) -> None:
    expected = payload.get("fingerprint")
    actual = manifest_fingerprint(payload)
    if expected != actual:
        raise ValueError(f"manifest fingerprint mismatch: stored={expected!r}, actual={actual!r}")
    kind = payload.get("kind")
    partitions = (payload.get("partitions", {}).keys()
                  if kind == "model1_experiment" else (None,))
    cross_partition: dict[tuple[str, str], str | None] = {}
    for partition in partitions:
        entries = manifest_entries(payload, partition=partition)
        paths: set[str] = set()
        commands: set[str] = set()
        for entry in entries:
            relative = entry.get("path")
            command = entry.get("command_key")
            digest = entry.get("content_sha256")
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
                raise ValueError(f"manifest entry has invalid relative path {relative!r}")
            if relative in paths:
                raise ValueError(f"duplicate sample path in manifest: {relative}")
            paths.add(relative)
            if not isinstance(command, str) or not command:
                raise ValueError(f"manifest entry {relative} has no command_key")
            if command in commands:
                raise ValueError(f"duplicate exact command in manifest partition: {command}")
            commands.add(command)
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise ValueError(f"manifest entry {relative} has no content SHA-256")
            if kind == "model1_experiment":
                for identity_type, identity in (
                    ("path", relative), ("command", command), ("content", digest),
                ):
                    identity_key = identity_type, identity
                    previous = cross_partition.setdefault(identity_key, partition)
                    if previous != partition:
                        raise ValueError(
                            f"experiment leakage by {identity_type}: {identity!r} occurs in "
                            f"{previous!r} and {partition!r}"
                        )


def read_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest {path} must contain a JSON object")
    validate_manifest(payload)
    return payload


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Seal and atomically write one manifest."""
    path = Path(path)
    sealed = seal_manifest(payload)
    validate_manifest(sealed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(sealed, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return sealed


def resolve_sample_paths(
    root: str | Path,
    payload: Mapping[str, Any],
    *,
    partition: str | None = None,
) -> tuple[Path, ...]:
    root = Path(root).resolve()
    resolved: list[Path] = []
    for entry in manifest_entries(payload, partition=partition):
        candidate = (root / entry["path"]).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"manifest path escapes corpus root: {entry['path']!r}") from exc
        if not candidate.is_dir():
            raise FileNotFoundError(f"manifest sample directory does not exist: {candidate}")
        resolved.append(candidate)
    return tuple(resolved)


def assert_zero_cohort_leakage(manifests: Iterable[Mapping[str, Any]]) -> None:
    """Reject sample, content, or exact-command reuse between frozen cohorts."""
    owners: dict[tuple[str, str], tuple[str, str]] = {}
    for payload in manifests:
        validate_manifest(payload)
        label = str(payload.get("cohort") or payload.get("name") or payload.get("fingerprint"))
        fingerprint = str(payload["fingerprint"])
        entries = manifest_entries(payload)
        for entry in entries:
            for identity_type, identity in (
                ("path", str(entry["path"])),
                ("content", str(entry["content_sha256"])),
                ("command", str(entry["command_key"])),
            ):
                key = identity_type, identity
                previous_fingerprint, previous_label = owners.setdefault(
                    key, (fingerprint, label)
                )
                if previous_fingerprint != fingerprint:
                    raise ValueError(
                        f"cohort leakage by {identity_type}: {identity!r} occurs in "
                        f"{previous_label!r} and {label!r}"
                    )
