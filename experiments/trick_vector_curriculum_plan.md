# TrickVector + Hybrid Scorer (vector-mode curriculum)

## Context

The previous plan in this file (now complete and committed) introduced the per-trick `Curriculum` system: each target trick gets a flat `rewards: {trick → reward}` dict in `curricula/<trick>.json`, scored by `Curriculum.score()`. That works, but it's hand-tuned per target — every new target trick needs its own dict, and there's no analytical relationship between tricks.

Asher's insight: tricks are **compositions of atomic mechanical components**. The board's motion decomposes into ~4 orthogonal axes:

| Axis            | What it captures                                          | Sign convention                       |
|-----------------|-----------------------------------------------------------|---------------------------------------|
| `body_rotation` | Person + board around vertical axis (FS/BS spin)          | `+1` per 180° BS, `−1` per 180° FS    |
| `shove_rotation`| Board around vertical axis, relative to person (shove-it) | `+1` per 180° BS shove, `−1` per FS   |
| `kickflip_axis` | Board around long axis                                    | `+1` per kickflip, `−1` per heelflip  |
| `dolphin_axis`  | Board around short axis                                   | `+1` dolphin, `+2` dragon             |

Plus a `stance` channel (`normal` / `FAKIE` / `SWITCH` / `NOLLIE` / `LATE`) for pose, orthogonal to mechanics.

Every trick is a vector in this space:
- `KICKFLIP = (0, 0, +1, 0)`
- `360 FLIP = (0, +2, +1, 0)` — adds a 360 BS shove to a kickflip
- `HARD FLIP = (0, −1, +1, 0)` — kickflip + FS pop shove-it
- `NIGHTMARE FLIP = (0, +1, +2, 0)` — pop shove + double kickflip
- `BACKSIDE FLIP = (+1, 0, +1, 0)` — BS 180 + kickflip
- `LASER FLIP = (0, −2, −1, 0)` — heelflip + FS 360 shove
- `DRAGON FLIP = (0, 0, 0, +2)` — two dolphin flips

A reward function over this space is **hand-engineered embedding + similarity metric** — the analytical version of what the parked PPO trick-conditioned net would learn from data. It collapses ~30 hand-tuned curricula into one scoring function plus per-trick overrides for the few cases where the geometry lies (e.g., 360 FLIP scores too high for a KICKFLIP target — drives basin drift).

The user explicitly retains the flat-dict scorer as a fallback: if the vector approach turns out to be wrong for a given trick, set `"scorer": "flat_dict"` in the curriculum and use the existing per-trick rewards dict. Vector becomes an **opt-in alternative**, not a replacement.

## Subsection 1 — `src/trueskate_ai/sim/trick_vector.py` (new module)

**Goal:** convert any normalised trick name → `TrickVector`, and score detected-vs-target with the hybrid metric.

Located under `sim/` (not `rl/cmaes/`) because it's a domain primitive — also useful for trick-library neighborhood search and (eventually) PPO embedding initialisation.

### `TrickVector` dataclass
```python
@dataclass(frozen=True)
class TrickVector:
    body_rotation: float = 0.0    # signed, units = half-rotations (180 = 1, 360 = 2, 540 = 3)
    shove_rotation: float = 0.0   # signed, same units; +1 = BS pop shove, −1 = FS pop shove
    kickflip_axis: float = 0.0    # signed, units = flips; +1 = kickflip, −1 = heelflip
    dolphin_axis: float = 0.0     # non-negative int; 1 = dolphin, 2 = dragon
    stance: str = "normal"        # "normal" | "FAKIE" | "SWITCH" | "NOLLIE" | "LATE"
```

### `BASE_VECTORS` — hand-curated atomic + named-composite vectors
~30 entries covering the irreducible atoms and the named composites whose decomposition is non-obvious from the name. Examples:
```python
BASE_VECTORS: dict[str, TrickVector] = {
    "OLLIE":            TrickVector(),
    "KICKFLIP":         TrickVector(kickflip_axis=+1),
    "HEELFLIP":         TrickVector(kickflip_axis=-1),
    "POP SHOVE-IT":     TrickVector(shove_rotation=+1),       # BS default
    "FS POP SHOVE-IT":  TrickVector(shove_rotation=-1),
    "BACKSIDE 180":     TrickVector(body_rotation=+1),
    "FRONTSIDE 180":    TrickVector(body_rotation=-1),
    "VARIAL KICKFLIP":  TrickVector(shove_rotation=+1, kickflip_axis=+1),
    "VARIAL HEELFLIP":  TrickVector(shove_rotation=-1, kickflip_axis=-1),  # varial heel = FS shove
    "INWARD HEELFLIP":  TrickVector(shove_rotation=+1, kickflip_axis=-1),  # BS shove + heel
    "NIGHTMARE FLIP":   TrickVector(shove_rotation=+1, kickflip_axis=+2),
    "HARD FLIP":        TrickVector(shove_rotation=-1, kickflip_axis=+1),
    "LASER FLIP":       TrickVector(shove_rotation=-2, kickflip_axis=-1),  # FS 360 shove + heel
    "DOLPHIN FLIP":     TrickVector(dolphin_axis=+1),
    "DRAGON FLIP":      TrickVector(dolphin_axis=+2),
    "BIG SPIN":         TrickVector(body_rotation=+2, shove_rotation=+1),  # BS 360 body + BS 180 board
    "BIGGER SPIN":      TrickVector(body_rotation=+3, shove_rotation=+1),  # BS 540 body + BS 180 board
    "GAZELLE SPIN":     TrickVector(body_rotation=+3, shove_rotation=+3),  # BS 540 body + BS 540 board
    "IMPOSSIBLE":       TrickVector(),  # rotation about back foot — outside our 4-axis model; treat as ollie-equivalent for now
    "MANUAL":           TrickVector(),
    ...
}
```

The user owns the canonical vector assignments per trick — Subagent A will produce a full draft from the `known_tricks.py` taxonomy and the user can adjust.

### `parse_trick_name(name) -> TrickVector | None`
Compositional parser, runs only when the name isn't directly in `BASE_VECTORS`:
```python
def parse_trick_name(name: str) -> TrickVector | None:
    tokens = normalize_trick_name(name).split()
    if not tokens: return None

    # 1. Stance modifier prefix
    stance = "normal"
    if tokens[0] in MODIFIERS:                     # FAKIE / SWITCH / NOLLIE / LATE
        stance = tokens.pop(0)

    # 2. Direct lookup of remaining string
    key = " ".join(tokens)
    if key in BASE_VECTORS:
        return replace(BASE_VECTORS[key], stance=stance)

    # 3. Compositional fallback — strip in this order:
    #    a. Direction prefix (BACKSIDE/FRONTSIDE) → sign on body_rotation
    #    b. Rotation level (180/360/540/720) → magnitude on body_rotation
    #    c. Composite marker (VARIAL/INWARD/HARD/DOLPHIN/etc) → handled by base lookup
    #    d. Multiplicity (DOUBLE/TRIPLE/QUAD) → multiplier on flip axis
    #    e. Base trick (FLIP→kickflip, HEEL/HEELFLIP→heelflip, etc)
    body_sign = +1
    if tokens and tokens[0] in DIRECTIONAL_MODIFIERS:
        body_sign = +1 if tokens[0] == "BACKSIDE" else -1
        tokens.pop(0)

    body_rot = 0
    if tokens and tokens[0] in {"180", "360", "540", "720"}:
        body_rot = body_sign * (int(tokens.pop(0)) // 180)
    elif tokens and tokens[0] in DIRECTIONAL_MODIFIERS:
        body_rot = body_sign * 1  # plain "BACKSIDE FLIP" implies 180

    multiplicity = 1
    for marker, n in [("DOUBLE", 2), ("TRIPLE", 3), ("QUAD", 4)]:
        if marker in tokens:
            multiplicity = n
            tokens.remove(marker)

    # Look up the residual base ("FLIP" / "HEEL" / "POP SHOVE-IT" / etc) in BASE_VECTORS
    residual_key = " ".join(tokens)
    base = BASE_VECTORS.get(residual_key)
    if base is None:
        return None  # unrecognised — caller handles (override or default 0.0)

    return TrickVector(
        body_rotation=body_rot or base.body_rotation,
        shove_rotation=base.shove_rotation,
        kickflip_axis=base.kickflip_axis * multiplicity,
        dolphin_axis=base.dolphin_axis * multiplicity,
        stance=stance,
    )
```
(Subagent A will refine — the order-of-operations needs careful testing across the family taxonomy. Strategy: parse, then verify by round-tripping every trick in `KNOWN_TRICKS` through the parser and asserting the result is non-`None` for tricks we expect to support.)

### `hybrid_score(target_v, detected_v, *, decay_rate, stance_penalty) -> float`
Tier × within-tier decay, per Q2:
```python
_AXES = ("body_rotation", "shove_rotation", "kickflip_axis", "dolphin_axis")

def hybrid_score(
    target_v: TrickVector,
    detected_v: TrickVector,
    *,
    decay_rate: float = 0.5,
    stance_penalty: float = 0.7,
    full_match_reward: float = 1.0,
) -> float:
    target_active = [a for a in _AXES if abs(getattr(target_v, a)) > 1e-6]

    if not target_active:
        # Target is OLLIE / MANUAL — penalize any extras
        l1 = sum(abs(getattr(detected_v, a)) for a in _AXES)
        base = full_match_reward / (1.0 + decay_rate * l1)
    else:
        # Tier: fraction of target axes where detected has same-sign nonzero
        matches = sum(
            1 for a in target_active
            if getattr(target_v, a) * getattr(detected_v, a) > 0
        )
        tier = matches / len(target_active)
        # Within-tier decay: L1 over ALL axes (catches magnitude misses + extras)
        l1 = sum(abs(getattr(target_v, a) - getattr(detected_v, a)) for a in _AXES)
        base = full_match_reward * tier / (1.0 + decay_rate * l1)

    # Stance penalty (asymmetric — only when target specifies a stance)
    if target_v.stance != "normal" and target_v.stance != detected_v.stance:
        base *= stance_penalty
    return base
```

**Worked examples** (decay_rate=0.5):

| Target            | Detected            | tier | L1 | decay | base reward |
|-------------------|---------------------|------|----|-------|-------------|
| KICKFLIP          | KICKFLIP            | 1.0  | 0  | 1.00  | **1.00**    |
| KICKFLIP          | 360 FLIP            | 1.0  | 2  | 0.50  | **0.50**    |
| KICKFLIP          | NIGHTMARE FLIP      | 1.0  | 2  | 0.50  | **0.50**    |
| KICKFLIP          | HARD FLIP           | 1.0  | 1  | 0.67  | **0.67**    |
| KICKFLIP          | POP SHOVE-IT        | 0    | 2  | 0.50  | **0.00**    |
| KICKFLIP          | HEELFLIP            | 0    | 2  | 0.50  | **0.00**    |
| 360 FLIP          | KICKFLIP            | 0.5  | 2  | 0.50  | **0.25**    |
| 360 FLIP          | 360 FLIP            | 1.0  | 0  | 1.00  | **1.00**    |
| BACKSIDE FLIP     | FRONTSIDE FLIP      | 0.5  | 2  | 0.50  | **0.25**    |
| NOLLIE KICKFLIP   | KICKFLIP            | 1.0  | 0  | 1.00  | **0.70** (stance miss) |
| KICKFLIP          | NOLLIE KICKFLIP     | 1.0  | 0  | 1.00  | **1.00** (target stance unspecified) |

Numbers reflect the math; user can adjust `decay_rate` per curriculum if these need to be more/less generous.

### `__main__` self-test
- Round-trip every trick in `KNOWN_TRICKS` through `parse_trick_name`; report names that parse to `None`.
- Score the table above and assert reward values within tolerance.
- Score ~10 known good/bad cases per atomic family.

## Subsection 2 — `Curriculum` extension

**File:** `src/trueskate_ai/rl/cmaes/curriculum.py` (extend, do not rewrite — the existing flat-dict path stays intact).

### Schema additions

| Field             | Type                          | Default       | Used when |
|-------------------|-------------------------------|---------------|-----------|
| `scorer`          | `"flat_dict"` \| `"vector"`   | `"flat_dict"` | both      |
| `overrides`       | `dict[str, float]`            | `{}`          | `scorer="vector"` |
| `vector_config`   | `{decay_rate, stance_penalty}`| defaults      | `scorer="vector"` |

`flat_dict`-mode curricula keep `rewards` and `default_reward` (untouched). `vector`-mode curricula use `overrides` + `vector_config` (and ignore `rewards`/`default_reward`).

### Schema example — vector mode
```json
{
  "target": "KICKFLIP",
  "warm_start": "trick_libraries/kickflip_2.json",
  "scorer": "vector",
  "vector_config": {
    "decay_rate": 0.5,
    "stance_penalty": 0.7
  },
  "overrides": {
    "360 FLIP": 0.0,
    "NIGHTMARE FLIP": 0.0
  },
  "failure_multiplier": "near_miss",
  "notes": "Vector-mode kickflip. Overrides zero out 360 FLIP and NIGHTMARE FLIP to prevent basin drift (vector would give ~0.5)."
}
```

### `Curriculum.score()` dispatch
```python
def score(self, result):
    if result is None: return 0.0
    components = [normalize_trick_name(c) for c in result.trick.split(" + ") if c.strip()]
    if not components: return 0.0

    if self.scorer == "vector":
        target_v = parse_trick_name(self.target)
        if target_v is None:
            raise ValueError(f"Curriculum target {self.target!r} could not be parsed to TrickVector")
        base = max(self._vector_score_component(c, target_v) for c in components)
    else:  # flat_dict (existing behaviour)
        base = max(self._flat_dict_score_component(c) for c in components)

    if result.status == "landed": return base
    return _resolve_failure_multiplier(self.failure_multiplier_spec)(base)

def _vector_score_component(self, component, target_v):
    if component in self.overrides:                         # explicit override wins
        return self.overrides[component]
    detected_v = parse_trick_name(component)
    if detected_v is None:
        return 0.0
    return hybrid_score(target_v, detected_v, **self.vector_config)
```

The existing flat-dict path renames internally:
- `_score_component` → `_flat_dict_score_component` (no behaviour change)
- `from_json` extended to read `scorer`, `overrides`, `vector_config` with defaults that preserve current behaviour

`from_json` also auto-includes `target → 1.0` in `overrides` for vector mode (parallel to the existing auto-include in `rewards`), so the user can omit it.

## Subsection 3 — Example vector-mode curriculum + journal entries

### New file: `curricula/kickflip_vector.json`
Vector-mode equivalent of `kickflip.json`, demonstrating the new schema. Asher can compare side-by-side and tune. Other tricks (360 flip, nightmare flip, etc.) stay on `flat_dict` until the vector approach is validated.

### `experiments/rl_poc_experiment_journal.md` (append)
Section dated **2026-05-11** — "TrickVector + hybrid scorer":
- Why: scaling the per-trick `rewards` dict didn't generalise — every new target trick needed manual tuning. Tricks have a real compositional structure; encoding it directly gives one scoring function for all targets.
- Vector model: 4 mechanical axes (body_rotation, shove_rotation, kickflip_axis, dolphin_axis) in rotation-count units (180=1, 360=2), plus stance categorical.
- Hybrid scorer: tier (axis-membership) × within-tier decay (L1 magnitude). Explicit per-trick overrides for cases where geometry lies.
- Curriculum gains opt-in `"scorer": "vector"` mode; flat_dict remains the default until vector is validated on real CMA-ES runs.
- Open: tuning `decay_rate` (currently 0.5); validating the hand-curated `BASE_VECTORS` table against real OCR detections; deciding whether `stance_penalty` should also penalise extras (target=normal, detected=NOLLIE).

### `experiments/rl_neural_net_experiment_journal.md` (append)
Section dated **2026-05-11** — "Hand-engineered trick embedding":
- The new TrickVector is the analytical version of the trick-conditioning embedding the parked PPO net would learn. Same data structure, hand-built.
- Future use: when PPO unparks, initialise the trick embedding head with these vectors (one row per trick), unfreeze for online updates. Cuts the cold-start by giving the embedding a structurally meaningful prior.
- The TrickVector is also a candidate index for trick-library nearest-neighbour search ("which library entry is closest to my target trick's vector?") — could enable cross-trick warm starts.

## Sub-agent execution plan

Two Sonnet sub-agents at high effort:

1. **Subagent A — TrickVector module + Curriculum extension + vector example** (Sonnet, high effort)
   - Scope: new `src/trueskate_ai/sim/trick_vector.py` (read `src/trueskate_ai/sim/known_tricks.py` for the trick taxonomy + frozensets); extend `src/trueskate_ai/rl/cmaes/curriculum.py` (read first, additive edit only — preserves existing flat_dict behaviour); new `curricula/kickflip_vector.json`.
   - Deliverable: TrickVector dataclass + BASE_VECTORS draft (~30 entries) + parser + hybrid_score; Curriculum `scorer`/`overrides`/`vector_config` fields + dispatch in `score()`; `__main__` self-tests passing on round-trip + the worked-example reward table.
   - Verify: `python -m trueskate_ai.sim.trick_vector` reports zero parser failures across `KNOWN_TRICKS` (or lists the failures so Asher can decide if they need BASE_VECTORS entries); `python -m trueskate_ai.rl.cmaes.curriculum` self-test still passes (existing flat_dict behaviour) AND the new vector-mode kickflip_vector.json loads + scores the worked-example table within tolerance.

2. **Subagent B — Journal entries** (Sonnet, high effort)
   - Scope: `experiments/rl_poc_experiment_journal.md`, `experiments/rl_neural_net_experiment_journal.md` only.
   - Deliverable: append-only entries per Subsection 3.
   - **Runs in parallel with A** — independent of code changes.

## Critical files

| Path | Change |
|------|--------|
| `src/trueskate_ai/sim/trick_vector.py` | **New** — `TrickVector`, `BASE_VECTORS`, `parse_trick_name`, `hybrid_score`, self-tests |
| `src/trueskate_ai/rl/cmaes/curriculum.py` | Extend — add `scorer`, `overrides`, `vector_config` fields; dispatch in `score()`; rename `_score_component` → `_flat_dict_score_component`; add `_vector_score_component` |
| `curricula/kickflip_vector.json` | **New** — vector-mode example of the kickflip curriculum |
| `experiments/rl_poc_experiment_journal.md` | Append TrickVector + hybrid-scorer entry (2026-05-11) |
| `experiments/rl_neural_net_experiment_journal.md` | Append hand-engineered embedding entry (2026-05-11) |

**Read-only references** (Subagent A consumes these):
- `src/trueskate_ai/sim/known_tricks.py` — full trick taxonomy + family frozensets + `MODIFIERS`/`DIRECTIONAL_MODIFIERS`
- `src/trueskate_ai/sim/trick_info_reader.py` — `TrickResult` shape
- `src/trueskate_ai/rl/reward.py` — `normalize_trick_name`, `near_miss_multiplier`

**Untouched** (no changes needed):
- `src/trueskate_ai/rl/reward.py` — fully covered by the previous refactor; vector-mode is purely a Curriculum-level dispatch
- `src/trueskate_ai/rl/cmaes/cmaes_optimizer.py` — already calls `curriculum.score(...)`; doesn't care which scorer mode
- `scripts/train/train_cmaes.py` — `--curriculum <path>` already accepts any curriculum schema
- `src/trueskate_ai/rl/device_worker.py`, `src/trueskate_ai/rl/ppo/collector.py` — already inject `scorer` callable; agnostic to internals
- All existing `curricula/*.json` — keep working unchanged (default `scorer: "flat_dict"`)

## Verification

1. **Static** — `python -m py_compile src/trueskate_ai/sim/trick_vector.py src/trueskate_ai/rl/cmaes/curriculum.py`
2. **Parser coverage** — `python -m trueskate_ai.sim.trick_vector` round-trips every entry in `KNOWN_TRICKS` (excluding `GRIND_SLIDE_FAMILY` which is outside the 4-axis model — those should return `None` cleanly, not crash). Reports any unexpected `None`s for Asher to triage.
3. **Hybrid scorer table** — `__main__` block in `trick_vector.py` asserts the worked-example reward table (KICKFLIP target × {KICKFLIP, 360 FLIP, NIGHTMARE FLIP, HARD FLIP, POP SHOVE-IT, HEELFLIP, NOLLIE KICKFLIP}) within ±0.01 tolerance.
4. **Curriculum backward-compat** — `python -m trueskate_ai.rl.cmaes.curriculum` self-test still passes for `kickflip.json` (flat_dict mode, existing behaviour unchanged).
5. **Curriculum vector-mode** — Extended `__main__` loads `curricula/kickflip_vector.json` and asserts: KICKFLIP=1.0, 360 FLIP=0.0 (override), NIGHTMARE FLIP=0.0 (override), HARD FLIP≈0.67 (vector default), POP SHOVE-IT=0.0 (no axis match), failed KICKFLIP = `near_miss(1.0) = 0.9`.
6. **CMA-ES smoke (deferred — needs device)** — `python scripts/train/train_cmaes.py --curriculum curricula/kickflip_vector.json --max-evals 2 --device-count 1` confirms end-to-end: vector-mode curriculum loads, `--initial-mean` defaults from warm_start, JSONL log records vector-scored rewards, no crashes.
