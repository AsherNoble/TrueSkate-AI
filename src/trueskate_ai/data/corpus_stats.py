"""Corpus statistics for the SLS frame→gesture corpus.

Walks sample dirs (``<root>/<session>/<park>/sample_*/`` — ``meta.json`` plus
optional ``.menu``/``.editor`` contamination markers) and aggregates what a
training-readiness call needs: kind mix, spin coverage, park/device balance,
contamination per kind, flick start-position coverage, Model-1/Model-2
trainable subsets, and collection rate.

STDLIB-ONLY by design: the identical module runs on the rig, the laptop, and
inside a bare Modal container walking the mounted ``trueskate-corpus`` volume
(``scripts/cloud/corpus_stats_modal.py``). CLI: ``scripts/data/corpus_stats.py``.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

# RL-safe gesture bounds for the start-position grid. Mirrored from
# sim/gestures.py (X_BOUND_MIN etc.) rather than imported: gestures.py pulls
# selenium, which the bare stats container doesn't ship.
_X_MIN, _X_MAX = 0.12, 1.0
_Y_MIN, _Y_MAX = 0.12, 0.88
GRID = 8

# Model 1 (trace extractor) trains on single-touch samples only: flicks, plus
# spin_flicks (drag + labelled spin-button hold). Everything clean feeds Model 2.
_M1_KINDS = ("flick", "spin_flick")

_SESSION_RE = re.compile(r"^(?P<device>.+?)_(?P<ts>\d{8}_\d{6})$")


def iter_samples(root: Path) -> Iterator[tuple[Path, dict | None, set[str], str]]:
    """Yield (sample_dir, meta|None, flags, session) for every sample under root.

    flags ⊆ {menu, editor, no_meta, bad_meta}. session is path-derived so it
    survives an unreadable meta; pass a corpus root or a single session dir.
    """
    root = Path(root)
    for sample_dir in sorted(root.rglob("sample_*")):
        if not sample_dir.is_dir():
            continue
        parts = sample_dir.relative_to(root).parts
        session = parts[0] if len(parts) >= 3 else root.name
        flags: set[str] = set()
        if (sample_dir / ".menu").exists():
            flags.add("menu")
        if (sample_dir / ".editor").exists():
            flags.add("editor")
        meta = None
        meta_path = sample_dir / "meta.json"
        if not meta_path.exists():
            flags.add("no_meta")
        else:
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                flags.add("bad_meta")
        yield sample_dir, meta, flags, session


def _bin(v: float, width: float) -> str:
    return f"{int(v / width) * width:.2f}"


def accumulate(samples: Iterable[tuple[Path, dict | None, set[str], str]]) -> dict:
    """Aggregate iter_samples() output into one JSON-serialisable stats dict."""
    n = 0
    flag_counts: Counter = Counter()
    kinds_all: Counter = Counter()
    kinds_clean: Counter = Counter()
    contam: dict[str, Counter] = defaultdict(Counter)
    per_session: dict[str, dict] = {}
    per_device_clean: Counter = Counter()
    per_park_clean: Counter = Counter()
    m1_per_park: Counter = Counter()
    spin_clean: Counter = Counter()
    grid = [[0] * GRID for _ in range(GRID)]
    dur_hist: Counter = Counter()
    ease_hist: Counter = Counter()
    reach_hist: Counter = Counter()
    hold_hist: Counter = Counter()
    frames_total = 0
    t_min = t_max = None

    for sample_dir, meta, flags, session in samples:
        n += 1
        for f in flags:
            flag_counts[f] += 1
        clean = not ({"menu", "editor"} & flags) and meta is not None
        sess = per_session.setdefault(
            session, {"n": 0, "clean": 0, "first_epoch_s": None, "last_epoch_s": None})
        sess["n"] += 1
        if meta is None:
            continue
        kind = str(meta.get("gesture_distribution", "?"))
        park = str(meta.get("park") or sample_dir.parent.name)
        # spin_active is authoritative when present; kind spin/spin_flick implies
        # a forced-on hold for samples predating the decoded spin fields.
        spin = bool(meta.get("spin_active", kind in ("spin", "spin_flick")))
        kinds_all[kind] += 1
        c = contam[kind]
        c["n"] += 1
        if "menu" in flags:
            c["menu"] += 1
        if "editor" in flags:
            c["editor"] += 1
        frames_total += int(meta.get("n_frames") or 0)
        t = meta.get("t_call_start_epoch_s")
        if t is not None:
            t = float(t)
            t_min = t if t_min is None else min(t_min, t)
            t_max = t if t_max is None else max(t_max, t)
            sess["first_epoch_s"] = t if sess["first_epoch_s"] is None else min(sess["first_epoch_s"], t)
            sess["last_epoch_s"] = t if sess["last_epoch_s"] is None else max(sess["last_epoch_s"], t)
        if not clean:
            continue
        sess["clean"] += 1
        kinds_clean[kind] += 1
        per_park_clean[park] += 1
        m = _SESSION_RE.match(session)
        per_device_clean[m.group("device") if m else session] += 1
        if spin:
            spin_clean[kind] += 1
            if meta.get("spin_hold_start_s") is not None and meta.get("spin_hold_end_s") is not None:
                hold_hist[_bin(float(meta["spin_hold_end_s"]) - float(meta["spin_hold_start_s"]), 0.2)] += 1
        if kind in _M1_KINDS and meta.get("waypoints"):
            m1_per_park[park] += 1
            wps = meta["waypoints"]
            x0, y0 = float(wps[0][0]), float(wps[0][1])
            gx = min(GRID - 1, max(0, int((x0 - _X_MIN) / (_X_MAX - _X_MIN) * GRID)))
            gy = min(GRID - 1, max(0, int((y0 - _Y_MIN) / (_Y_MAX - _Y_MIN) * GRID)))
            grid[gy][gx] += 1
            if meta.get("duration") is not None:
                dur_hist[_bin(float(meta["duration"]), 0.05)] += 1
            if meta.get("easing_power") is not None:
                ease_hist[_bin(float(meta["easing_power"]), 0.25)] += 1
            xn, yn = float(wps[-1][0]), float(wps[-1][1])
            reach_hist[_bin(((xn - x0) ** 2 + (yn - y0) ** 2) ** 0.5, 0.05)] += 1

    clean_total = sum(kinds_clean.values())
    m1_total = sum(m1_per_park.values())
    occupied = sum(1 for row in grid for v in row if v > 0)
    span_days = 0.0
    if t_min is not None and t_max is not None and t_max > t_min:
        span_days = (t_max - t_min) / 86400.0
    return {
        "generated_epoch_s": time.time(),
        "totals": {
            "samples": n,
            "clean": clean_total,
            "menu": flag_counts.get("menu", 0),
            "editor": flag_counts.get("editor", 0),
            "no_meta": flag_counts.get("no_meta", 0),
            "bad_meta": flag_counts.get("bad_meta", 0),
            "frames": frames_total,
        },
        "kinds_all": dict(kinds_all),
        "kinds_clean": dict(kinds_clean),
        "spin_clean_by_kind": dict(spin_clean),
        "spin_clean_total": sum(spin_clean.values()),
        "contamination_by_kind": {k: dict(v) for k, v in contam.items()},
        "model1_trainable": {
            "total": m1_total,
            "spin": spin_clean.get("spin_flick", 0),
            "per_park": dict(m1_per_park),
        },
        "model2_usable": clean_total,
        "per_device_clean": dict(per_device_clean),
        "per_park_clean": dict(per_park_clean),
        "per_session": per_session,
        "flick_start_grid": grid,
        "flick_start_grid_occupied": f"{occupied}/{GRID * GRID}",
        "flick_duration_hist": dict(dur_hist),
        "flick_easing_hist": dict(ease_hist),
        "flick_reach_hist": dict(reach_hist),
        "spin_hold_len_hist": dict(hold_hist),
        "span_days": round(span_days, 2),
        "samples_per_day": round(n / span_days, 1) if span_days > 0 else None,
    }


def merge(a: dict, b: dict) -> dict:
    """Merge two accumulate() outputs (e.g. Modal volume + rig-local pending).

    Counter-like sections add; grids add cellwise; span re-derives from the
    union of session first/last stamps.
    """
    def _addc(x: dict, y: dict) -> dict:
        out = Counter(x)
        out.update(y)
        return dict(out)

    out = json.loads(json.dumps(a))  # deep copy, keeps it plainly serialisable
    for key in ("kinds_all", "kinds_clean", "spin_clean_by_kind", "per_device_clean",
                "per_park_clean", "flick_duration_hist", "flick_easing_hist",
                "flick_reach_hist", "spin_hold_len_hist"):
        out[key] = _addc(a.get(key, {}), b.get(key, {}))
    for key in ("totals",):
        out[key] = _addc(a.get(key, {}), b.get(key, {}))
    out["spin_clean_total"] = a.get("spin_clean_total", 0) + b.get("spin_clean_total", 0)
    out["model2_usable"] = a.get("model2_usable", 0) + b.get("model2_usable", 0)
    out["contamination_by_kind"] = {
        k: _addc(a.get("contamination_by_kind", {}).get(k, {}),
                 b.get("contamination_by_kind", {}).get(k, {}))
        for k in {*a.get("contamination_by_kind", {}), *b.get("contamination_by_kind", {})}
    }
    out["model1_trainable"] = {
        "total": a["model1_trainable"]["total"] + b["model1_trainable"]["total"],
        "spin": a["model1_trainable"]["spin"] + b["model1_trainable"]["spin"],
        "per_park": _addc(a["model1_trainable"]["per_park"], b["model1_trainable"]["per_park"]),
    }
    out["per_session"] = {**a.get("per_session", {}), **b.get("per_session", {})}
    out["flick_start_grid"] = [
        [av + bv for av, bv in zip(ar, br)]
        for ar, br in zip(a["flick_start_grid"], b["flick_start_grid"])
    ]
    occupied = sum(1 for row in out["flick_start_grid"] for v in row if v > 0)
    out["flick_start_grid_occupied"] = f"{occupied}/{GRID * GRID}"
    stamps = [s[k] for s in out["per_session"].values()
              for k in ("first_epoch_s", "last_epoch_s") if s.get(k) is not None]
    span_days = (max(stamps) - min(stamps)) / 86400.0 if len(stamps) >= 2 else 0.0
    out["span_days"] = round(span_days, 2)
    n = out["totals"]["samples"]
    out["samples_per_day"] = round(n / span_days, 1) if span_days > 0 else None
    return out


def _top(d: dict, k: int = 6) -> str:
    items = sorted(d.items(), key=lambda kv: -kv[1])[:k]
    return " | ".join(f"{name} {count:,}" for name, count in items) or "—"


def _hist_line(h: dict) -> str:
    return " ".join(f"{k}:{v:,}" for k, v in sorted(h.items(), key=lambda kv: float(kv[0]))) or "—"


def _contam_line(c: dict) -> str:
    parts = []
    for kind, d in sorted(c.items()):
        nn = max(1, d.get("n", 0))
        parts.append(f"{kind} ed {100 * d.get('editor', 0) / nn:.1f}%"
                     f"/menu {100 * d.get('menu', 0) / nn:.1f}%")
    return " | ".join(parts) or "—"


def summarize(stats: dict) -> str:
    """Terse human-readable digest of an accumulate()/merge() stats dict."""
    t = stats["totals"]
    n = max(1, t["samples"])
    return "\n".join([
        f"samples: {t['samples']:,} | clean {stats['model2_usable']:,} "
        f"({100 * stats['model2_usable'] / n:.1f}%) | menu {t['menu']:,} | editor {t['editor']:,} "
        f"| meta missing/bad {t.get('no_meta', 0) + t.get('bad_meta', 0):,}",
        f"kinds (clean): {_top(stats['kinds_clean'], 8)}",
        f"spin-active (clean): {stats['spin_clean_total']:,} ({_top(stats['spin_clean_by_kind'], 5)})",
        f"Model 1 trainable (clean flick+spin_flick): {stats['model1_trainable']['total']:,} "
        f"[spin_flick: {stats['model1_trainable']['spin']:,}]",
        f"Model 2 usable (all clean): {stats['model2_usable']:,}",
        f"devices (clean): {_top(stats['per_device_clean'], 4)}",
        f"parks (clean): {_top(stats['per_park_clean'], 6)}",
        f"flick start coverage: {stats['flick_start_grid_occupied']} grid cells occupied",
        f"flick duration hist: {_hist_line(stats['flick_duration_hist'])}",
        f"flick reach hist: {_hist_line(stats['flick_reach_hist'])}",
        f"spin hold-length hist: {_hist_line(stats['spin_hold_len_hist'])}",
        f"contamination: {_contam_line(stats['contamination_by_kind'])}",
        f"span: {stats['span_days']} days, ~{stats['samples_per_day'] or '?'} samples/day "
        f"(whole-span avg); frames: {t['frames']:,}; sessions: {len(stats['per_session'])}",
    ])
