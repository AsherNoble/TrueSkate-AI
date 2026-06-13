"""Per-target-trick curriculum for CMA-ES reward shaping.

A Curriculum is a flat trick→reward dict (loaded from JSON) plus a default
reward for unlisted recognised tricks and a togglable failure multiplier.
It also carries the gesture-vector dimensionality for CMA-ES (``num_gestures``)
and optional per-slot initial means / inter-gesture delays. When
``num_gestures != 2`` the curriculum MUST supply either explicit priors or
a ``warm_start`` whose gesture count matches.

Curriculum JSONs live in `curricula/<trick>.json` at the repo root.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from trueskate_ai.rl.reward import near_miss_multiplier, normalize_trick_name
from trueskate_ai.sim.known_tricks import KNOWN_TRICKS
from trueskate_ai.sim.trick_info_reader import TrickResult

_PARAMS_PER_SLOT = 8  # mirrors action_param.PARAMS_PER_SLOT (kept local to avoid import cycle)


def _resolve_failure_multiplier(spec) -> Callable[[float], float]:
    """Convert the JSON `failure_multiplier` field to a callable.

    Accepts:
        "near_miss" — max(0, base * (base - 0.1))   (default)
        "zero"     — always 0
        float K    — constant multiplier K (clamped >= 0)
    """
    if spec == "near_miss" or spec is None:
        return near_miss_multiplier
    if spec == "zero":
        return lambda base: 0.0
    if isinstance(spec, (int, float)):
        k = max(0.0, float(spec))
        return lambda base: max(0.0, base) * k
    raise ValueError(f"Unrecognised failure_multiplier: {spec!r}")


@dataclass(frozen=True)
class Curriculum:
    target: str
    rewards: dict[str, float]
    default_reward: float = 0.0
    warm_start: Path | None = None
    failure_multiplier_spec: str | float = "near_miss"
    notes: str = ""
    num_gestures: int = 2
    use_spin: bool = False
    initial_means: tuple[tuple[float, ...], ...] | None = None
    initial_delays: tuple[float, ...] | None = None

    @classmethod
    def from_json(cls, path: Path) -> "Curriculum":
        path = Path(path)
        data = json.loads(path.read_text())
        target = normalize_trick_name(data["target"])
        if not target:
            raise ValueError(f"Curriculum {path} missing 'target'")
        rewards = {normalize_trick_name(k): float(v) for k, v in data.get("rewards", {}).items()}
        # Auto-include the target with reward 1.0 if not explicitly listed.
        rewards.setdefault(target, 1.0)
        warm_start_raw = data.get("warm_start")
        warm_start = (path.parent.parent / warm_start_raw).resolve() if warm_start_raw else None
        # Allow absolute paths in the JSON too:
        if warm_start_raw and Path(warm_start_raw).is_absolute():
            warm_start = Path(warm_start_raw)

        num_gestures = int(data.get("num_gestures", 2))
        if num_gestures < 1:
            raise ValueError(
                f"Curriculum {path}: num_gestures must be >= 1, got {num_gestures}"
            )

        raw_means = data.get("initial_means")
        if raw_means is not None:
            if not isinstance(raw_means, list) or len(raw_means) != num_gestures:
                got = len(raw_means) if isinstance(raw_means, list) else type(raw_means).__name__
                raise ValueError(
                    f"Curriculum {path}: initial_means must be a list of {num_gestures} "
                    f"per-slot entries, got {got}"
                )
            initial_means: tuple[tuple[float, ...], ...] | None = tuple(
                tuple(float(v) for v in slot) for slot in raw_means
            )
            for i, slot in enumerate(initial_means):
                if len(slot) != _PARAMS_PER_SLOT:
                    raise ValueError(
                        f"Curriculum {path}: initial_means[{i}] must have exactly "
                        f"{_PARAMS_PER_SLOT} values (3 waypoints * 2 coords + duration "
                        f"+ easing), got {len(slot)}"
                    )
        else:
            initial_means = None

        raw_delays = data.get("initial_delays")
        expected_delays = max(0, num_gestures - 1)
        if raw_delays is not None:
            if not isinstance(raw_delays, list) or len(raw_delays) != expected_delays:
                got = len(raw_delays) if isinstance(raw_delays, list) else type(raw_delays).__name__
                raise ValueError(
                    f"Curriculum {path}: initial_delays must be a list of "
                    f"{expected_delays} values (num_gestures - 1), got {got}"
                )
            initial_delays: tuple[float, ...] | None = tuple(float(v) for v in raw_delays)
        else:
            initial_delays = None

        # When num_gestures != 2 there is no default prior — fail fast if the
        # curriculum supplies neither warm_start nor embedded priors.
        if num_gestures != 2 and warm_start is None:
            missing = []
            if initial_means is None:
                missing.append("initial_means")
            if expected_delays > 0 and initial_delays is None:
                missing.append("initial_delays")
            if missing:
                raise ValueError(
                    f"Curriculum {path}: num_gestures={num_gestures} requires either "
                    f"'warm_start' or {' and '.join(missing)} to seed CMA-ES"
                )

        return cls(
            target=target,
            rewards=rewards,
            default_reward=float(data.get("default_reward", 0.0)),
            warm_start=warm_start,
            failure_multiplier_spec=data.get("failure_multiplier", "near_miss"),
            notes=data.get("notes", ""),
            num_gestures=num_gestures,
            use_spin=bool(data.get("use_spin", False)),
            initial_means=initial_means,
            initial_delays=initial_delays,
        )

    def _score_component(self, component: str) -> float:
        if component in self.rewards:
            return self.rewards[component]
        if component in KNOWN_TRICKS:
            return self.default_reward
        return 0.0

    def score(self, result: TrickResult | None) -> float:
        """Score a TrickResult using flat-dict lookup. Combos: max-component wins."""
        if result is None:
            return 0.0
        components = [normalize_trick_name(c) for c in result.trick.split(" + ") if c.strip()]
        if not components:
            return 0.0
        base = max(self._score_component(c) for c in components)
        if result.status == "landed":
            return base
        # failed / unknown
        multiplier_fn = _resolve_failure_multiplier(self.failure_multiplier_spec)
        return multiplier_fn(base)


# --- Self-test -------------------------------------------------------------
if __name__ == "__main__":
    import sys
    repo_root = Path(__file__).resolve().parents[4]
    curr_path = repo_root / "curricula" / "kickflip.json"
    if not curr_path.exists():
        print(f"FAIL: curriculum {curr_path} missing")
        sys.exit(1)
    c = Curriculum.from_json(curr_path)
    cases = [
        (TrickResult("KICKFLIP", "landed"),       1.0),
        (TrickResult("KICKFLIP", "failed"),       0.9),     # near_miss(1.0) = 0.9
        (TrickResult("VARIAL KICKFLIP", "landed"), c.rewards.get("VARIAL KICKFLIP", c.default_reward)),
        (TrickResult("OLLIE", "landed"),          c.default_reward),  # in KNOWN_TRICKS, not in rewards
        (TrickResult("BLATANT NONSENSE", "landed"), 0.0),
        (None,                                     0.0),
        (TrickResult("KICKFLIP + 50 50 GRIND", "landed"), 1.0),  # combo max
    ]
    ok = True
    for result, expected in cases:
        actual = c.score(result)
        status = "PASS" if abs(actual - expected) < 1e-6 else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"  [{status}] score({result!r}) = {actual:.3f} (expected {expected:.3f})")
    print("Curriculum tests OK" if ok else "FAILURES")
    sys.exit(0 if ok else 1)
