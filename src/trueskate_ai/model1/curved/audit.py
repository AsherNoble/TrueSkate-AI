"""Frozen 100-clip human audit scoring. Uncertainty never counts as a pass."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np

from .contract import SCHEMA, digest, save_json
from .capture import file_sha256


def verify_manifest(manifest):
    if manifest.get("schema") != SCHEMA or manifest.get("sha256") != digest(
            {k: v for k, v in manifest.items() if k != "sha256"}):
        raise ValueError("invalid curved manifest fingerprint")


def score_review(manifest, report):
    verify_manifest(manifest)
    if (report.get("manifest_sha256") != manifest["sha256"] or
            report.get("round") != manifest["round"]):
        raise ValueError("review belongs to a different frozen audit")
    rows = report.get("samples", [])
    ids = [r.get("id") for r in rows]
    wanted = [s["id"] for s in manifest["samples"]]
    if len(ids) != 100 or len(set(ids)) != 100 or set(ids) != set(wanted):
        raise ValueError("review must cover exactly all 100 unique clips")
    indexed = {r["id"]: r for r in rows}
    counts = dict(meet=0, fail=0, unclear=0)
    reasons, gross = {}, []
    for sample in manifest["samples"]:
        row = indexed[sample["id"]]
        rating = row.get("rating")
        if rating not in counts or type(row.get("systematic")) is not bool:
            raise ValueError("every clip needs a rating and an explicit systemic-defect flag")
        reason = row.get("reason", "").strip()
        if rating != "meet" and not reason:
            raise ValueError("failed/unclear clips require a reason")
        if row["systematic"]:
            if not reason:
                raise ValueError("gross/systematic defects require a reason")
            gross.append(sample["id"])
        if rating == "meet":
            times = np.asarray(sample["frame_times_s"])
            # Release marks are optional since 2026-10-12 (operator decision); a supplied one is checked.
            released = row.get("release_frame") is not None
            marks = [("onset_frame", "onset_s")] + ([("release_frame", "release_s")] if released else [])
            for key, target in marks:
                frame = row.get(key)
                if type(frame) is not int or not 0 <= frame < len(times):
                    raise ValueError(f"passing clips require an observed onset frame mark ({key} invalid)")
                expected = int(np.argmin(abs(times-sample[target])))
                if abs(frame-expected) > 1:
                    rating = "fail"
                    reason = "observed contact boundary differs by more than one native frame"
            if released and row["release_frame"] <= row["onset_frame"]:
                raise ValueError("release must follow onset")
        counts[rating] += 1
        if reason:
            reasons[str(sample["id"])] = reason
    return dict(round=manifest["round"], manifest_sha256=manifest["sha256"],
                counts=counts, gross_or_systematic=gross, reasons=reasons,
                passed=counts["meet"] >= 95 and not gross)


def freeze_dataset(ledger):
    """Whole sessions define partitions; seeded selection occurs within each park/split."""
    qualified = ledger.get("qualified")
    if not qualified:
        raise ValueError("qualification required")
    entries = []
    pool_count = 0
    seen_paths, seen_ids = set(), set()
    for park in ledger.definition["parks"]:
        sessions = ledger.sessions(phase="production", park=park)
        if len(sessions) != 10:
            raise ValueError("ten production sessions per park required")
        candidates = {s: [] for s in ("train", "validation", "test")}
        for session in sessions:
            samples = ledger.admitted(session["id"])
            if len(samples) != 280 or any(ledger.remaining(session["id"], s, include_pending=False)
                                           for s in ("random_chain", "arc")):
                raise ValueError("complete the separate 14,000-clip production pool first")
            pool_count += len(samples)
            for row in samples:
                path = Path(row["sample_path"])
                if path in seen_paths or row["id"] in seen_ids:
                    raise ValueError("duplicate production sample")
                seen_paths.add(path)
                seen_ids.add(row["id"])
                meta = json.loads((path/"meta.json").read_text())
                if (meta["fingerprint"] != qualified["fingerprint"] or meta["phase"] != "production"
                        or meta["session_id"] != session["id"] or meta["split"] != session["split"]
                        or meta["park"] != park or meta["candidate_id"] != row["id"]
                        or not meta["admission"]["accepted"]):
                    raise ValueError("production provenance changed")
                if any((path/m).exists() for m in (".menu", ".trace_mismatch", ".contaminated")):
                    raise ValueError("admitted pool contains a contamination marker")
                candidates[session["split"]].append(dict(
                    id=row["id"], sample_dir=str(path.relative_to(ledger.root)),
                    park=park, session=session["id"], recording=meta["source_recording_id"],
                    split=session["split"], meta_sha256=file_sha256(path/"meta.json"),
                    media_sha256=file_sha256(path/"clip.mp4")))
        for split, count in (("train", 2000), ("validation", 250), ("test", 250)):
            seed = int(digest([ledger.definition["seed"], park, split])[:16], 16)
            entries.extend(random.Random(seed).sample(candidates[split], count))
    if pool_count != 14000 or len(entries) != 12500:
        raise ValueError("production count mismatch")
    result = dict(schema=SCHEMA, kind="curved-training", definition=ledger.definition,
                  qualified=qualified, pool_count=pool_count, samples=entries)
    result["sha256"] = digest(result)
    destination = ledger.root/"dataset.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise ValueError("frozen dataset changed")
    else:
        save_json(destination, result)
    return result


def freeze_human(source: Path, output: Path, parks):
    """Freeze ten operator-selected, single-contact expert clips without gesture labels."""
    from .capture import probe_pts
    rows = json.loads(Path(source).read_text())
    if len(rows) != 10 or any(sum(r["park"] == p for r in rows) != 2 for p in parks):
        raise ValueError("exactly two human clips from each selected park required")
    samples = []
    for row in rows:
        if row.get("single_contact") is not True or row.get("in_scope") is not True:
            raise ValueError("confirm each clip is one contact within MVP scope")
        path = Path(row["video"]).resolve()
        pts = probe_pts(path)
        if len(pts) != 120:
            raise ValueError("human clips must contain 120 native 60 Hz frames")
        samples.append(dict(park=row["park"], video=str(path), sha256=file_sha256(path),
                            frame_times_s=(pts-pts[0]).tolist()))
    value = dict(schema=SCHEMA, kind="frozen-human-qualitative", samples=samples)
    value["sha256"] = digest(value)
    save_json(output, value)
    return value
