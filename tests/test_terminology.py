"""Current docs and the renamed code use GESTURES.md's vocabulary (its "Terminology" section).

Docs: the stop-using words never appear in prose. Code spans, code blocks and link
targets are names, not prose, so they are skipped, as are GESTURES.md's own "Stop
using" and "Legacy names" tables. The journal is checked from 2026-10-14, when the
vocabulary started to apply.

Code renamed on 2026-10-16: no name or word contains "stroke" or "piece", except the
legacy names that saved data and checkpoints still use (GESTURES.md, "Legacy names").
Those may appear only as quoted strings or as keyword names that become saved keys.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROSE_BANNED = re.compile(
    r"\b(strokes?|pieces?|swipe[sd]?|swiping|drag(?:s|ged|ging)?|traces?|kinks?|touch contacts?)\b", re.I)
CURRENT_DOCS = ("GESTURES.md", "AGENTS.md", "README.md", "DEPLOYMENT.md", "research/SYSTEM_MAP.md",
                "research/protocols/trick_speed_batch.md", "research/protocols/model1_curved_collection_v1.md",
                "docs/CURVED_MODEL1.md", "docs/WORKFLOWS.md", "docs/XR_CONTROL.md")
JOURNAL_SINCE = "2026-10-14"
SKIPPED_GESTURES_SECTIONS = ("### Stop using", "### Legacy names")

CODE_BANNED = re.compile(r"[A-Za-z_]*(?:stroke|piece)[A-Za-z_]*", re.I)
RENAMED_CODE = (
    "src/trueskate_ai/model1/trick", "src/trueskate_ai/model2",
    "src/trueskate_ai/research/kickflip_repeatability.py", "src/trueskate_ai/research/kickflip_tolerance.py",
    "src/trueskate_ai/research/kickflip_review.py", "src/trueskate_ai/control/hid_pointer.py",
    "scripts/collection/trick_speed_batch.py", "scripts/collection/probe_kickflip_repeatability.py",
    "scripts/collection/probe_kickflip_tolerance.py", "scripts/collection/run_pointer_schedule.py",
    "scripts/model1/train_trick_modal.py", "scripts/model2",
    "tests/test_trick_speed_batch.py", "tests/test_trick_speed_cli.py", "tests/test_trick_model1.py",
    "tests/test_kickflip_repeatability.py", "tests/test_kickflip_tolerance.py",
    "tests/test_model2_causal_timeline.py", "tests/test_bc_multitouch_tracking.py",
    "tests/test_hid_pointer.py", "tests/test_pointer_schedule_safety.py",
)
LEGACY_NAMES = {
    # model1_trick_v1 plans, executions, admitted clips, manifests, summaries and results
    "strokes", "stroke", "stroke_timing", "stroke_visible_onsets_s", "stroke_visible_lags_s",
    "stroke_visible_lag_s", "sent_strokes", "planned_strokes", "strokes_sent", "strokes_dropped",
    "stroke_families", "stroke_ms", "labelled_strokes", "predicted_strokes", "extra_strokes", "pieces",
    # Model 2 checkpoints saved before 2026-10-16
    "stroke_dim", "stroke_embed", "stroke_pos", "stroke_head",
}


def _prose(text: str) -> str:
    """Markdown with code blocks, code spans and link targets removed."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    return re.sub(r"\]\([^)]*\)", "]", text)


def _without_sections(text: str, headings) -> str:
    kept, skipping = [], False
    for line in text.splitlines():
        if line.startswith("#"):
            skipping = any(line.startswith(h) for h in headings)
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def _journal_since(text: str, since: str) -> str:
    entries = re.split(r"(?m)^(?=## )", text)
    return "\n".join(e for e in entries if not e.startswith("## ") or e[3:13] >= since)


def _violations(name: str, text: str) -> list[str]:
    found = []
    for number, line in enumerate(_prose(text).splitlines(), 1):
        found += [f"{name}:{number}: {m.group(0)!r}" for m in PROSE_BANNED.finditer(line)]
    return found


@pytest.mark.parametrize("name", CURRENT_DOCS)
def test_current_docs_use_the_vocabulary(name):
    text = (ROOT / name).read_text(encoding="utf-8")
    if name == "GESTURES.md":
        text = _without_sections(text, SKIPPED_GESTURES_SECTIONS)
    assert not _violations(name, text)


def test_recent_journal_entries_use_the_vocabulary():
    text = _journal_since((ROOT / "research/JOURNAL.md").read_text(encoding="utf-8"), JOURNAL_SINCE)
    assert not _violations("research/JOURNAL.md", text)


def _code_files():
    for entry in RENAMED_CODE:
        path = ROOT / entry
        yield from sorted(path.glob("*.py")) if path.is_dir() else (path,)


def _code_violations(path: Path) -> list[str]:
    found = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in CODE_BANNED.finditer(line):
            word, before, after = m.group(0), line[m.start() - 1:m.start()], line[m.end():m.end() + 2]
            quoted = before in "\"'" and before and after[:1] == before
            keyword = after[:1] == "=" and after[1:2] != "="
            if word in LEGACY_NAMES and (quoted or keyword):
                continue
            found.append(f"{path.relative_to(ROOT)}:{number}: {word!r}")
    return found


def test_renamed_code_uses_gesture_and_segment():
    files = list(_code_files())
    assert files and all(f.exists() for f in files)
    violations = [v for f in files for v in _code_violations(f)]
    assert not violations, "\n".join(violations)


def test_the_checker_catches_the_old_words():
    assert _violations("x", "Each stroke has two pieces, like a swipe or a trace.") == [
        "x:1: 'stroke'", "x:1: 'pieces'", "x:1: 'swipe'", "x:1: 'trace'"]
    assert not _violations("x", "Names stay: `extract_demo_strokes.py` and [link](a_trace.md).")
