"""Canonical True Skate trick taxonomy — transcribed from in-game trick book (68% coverage).

Single source of truth for:
  - OCR fuzzy matching (`KNOWN_TRICKS` — broad union, used by trick_info_reader)
  - Reward family classification (`*_FAMILY` frozensets + `mechanical_family()`,
    used by the CMA-ES curriculum system)

Modifiers (FAKIE, SWITCH, NOLLIE, LATE) prefix base tricks; they are not standalone
trick names.
"""

# ---------------------------------------------------------------------------
# Modifier sets (not standalone tricks)
# ---------------------------------------------------------------------------

MODIFIERS: frozenset[str] = frozenset({"FAKIE", "SWITCH", "NOLLIE", "LATE"})

# ---------------------------------------------------------------------------
# Trick families
# ---------------------------------------------------------------------------

# Kickflip-mechanic tricks: kickflip component is the flip mechanic.
# Includes varial kickflip / nightmare flip, hard flip, 360/540/720 flip series,
# and BS/FS flip variants.
KICKFLIP_FAMILY: frozenset[str] = frozenset(
    {
        # --- Base kickflips ---
        "KICKFLIP",
        "DOUBLE KICKFLIP",
        "TRIPLE KICKFLIP",
        "QUAD KICKFLIP",
        # --- Varial kickflips (kickflip + pop shove-it) ---
        "VARIAL KICKFLIP",
        "NIGHTMARE FLIP", # name for varial double flip
        "VARIAL TRIPLE KICKFLIP",
        "VARIAL QUAD KICKFLIP",
        # --- 360 flip series ---
        "360 FLIP",
        "360 DOUBLE FLIP",
        "360 TRIPLE FLIP",
        "360 QUAD FLIP",
        # --- 540 flip series ---
        "540 FLIP",
        "540 DOUBLE FLIP",
        "540 TRIPLE FLIP",
        "540 QUAD FLIP",
        # --- 720 flip series ---
        "720 FLIP",
        "720 DOUBLE FLIP",
        "720 TRIPLE FLIP",
        "720 QUAD FLIP",
        # --- Hard flip series (kickflip + FS pop shove-it) ---
        "HARD FLIP",
        "DOUBLE HARD FLIP",
        "TRIPLE HARD FLIP",
        "QUAD HARD FLIP",
        # --- 360 Hard flip series ---
        "360 HARD FLIP",
        "360 DOUBLE HARD FLIP",
        "360 TRIPLE HARD FLIP",
        "360 QUAD HARD FLIP",
        # --- 540 Hard flip series ---
        "540 HARD FLIP",
        "540 DOUBLE HARD FLIP",
        "540 TRIPLE HARD FLIP",
        "540 QUAD HARD FLIP",
        # --- 720 Hard flip series ---
        "720 HARD FLIP",
        "720 DOUBLE HARD FLIP",
        "720 TRIPLE HARD FLIP",
        "720 QUAD HARD FLIP",
        # --- Backside flip series (BS 180 + kickflip) ---
        "BACKSIDE FLIP",
        "BACKSIDE DOUBLE FLIP",
        "BACKSIDE TRIPLE FLIP",
        "BACKSIDE QUAD FLIP",
        # --- Frontside flip series (FS 180 + kickflip) ---
        "FRONTSIDE FLIP",
        "FRONTSIDE DOUBLE FLIP",
        "FRONTSIDE TRIPLE FLIP",
        "FRONTSIDE QUAD FLIP",
        # --- Backside 360 flip series ---
        "BACKSIDE 360 FLIP",
        "BACKSIDE 360 DOUBLE FLIP",
        "BACKSIDE 360 TRIPLE FLIP",
        "BACKSIDE 360 QUAD FLIP",
        # --- Frontside 360 flip series ---
        "FRONTSIDE 360 FLIP",
        "FRONTSIDE 360 DOUBLE FLIP",
        "FRONTSIDE 360 TRIPLE FLIP",
        "FRONTSIDE 360 QUAD FLIP",
        # --- Backside 540 flip series ---
        "BACKSIDE 540 FLIP",
        "BACKSIDE 540 DOUBLE FLIP",
        "BACKSIDE 540 TRIPLE FLIP",
        "BACKSIDE 540 QUAD FLIP",
        # --- Frontside 540 flip series ---
        "FRONTSIDE 540 FLIP",
        "FRONTSIDE 540 DOUBLE FLIP",
        "FRONTSIDE 540 TRIPLE FLIP",
        "FRONTSIDE 540 QUAD FLIP",
        # --- Backside 720 flip series ---
        "BACKSIDE 720 FLIP",
        "BACKSIDE 720 DOUBLE FLIP",
        "BACKSIDE 720 TRIPLE FLIP",
        "BACKSIDE 720 QUAD FLIP",
        # --- Frontside 720 flip series ---
        "FRONTSIDE 720 FLIP",
        "FRONTSIDE 720 DOUBLE FLIP",
        "FRONTSIDE 720 TRIPLE FLIP",
        "FRONTSIDE 720 QUAD FLIP",
    }
)

# Heelflip-mechanic tricks: heelflip component is the flip mechanic.
# Includes varial heelflip (with laser names for the 360 varial forms),
# inward heelflip (heelflip + BS pop shove-it), and BS/FS heel variants.
HEELFLIP_FAMILY: frozenset[str] = frozenset(
    {
        # --- Base heelflips ---
        "HEELFLIP",
        "DOUBLE HEELFLIP",
        "TRIPLE HEELFLIP",
        "QUAD HEELFLIP",
        # --- Varial heelflip series (heelflip + FS pop shove-it) ---
        "VARIAL HEELFLIP",
        "VARIAL DOUBLE HEELFLIP",
        "VARIAL TRIPLE HEELFLIP",
        "VARIAL QUAD HEELFLIP",
        # --- Laser flip series (heelflip + FS 360 pop shove-it) ---
        "LASER FLIP",
        "DOUBLE LASER FLIP",
        "TRIPLE LASER FLIP",
        "QUAD LASER FLIP",
        # --- 540 Varial heelflip series ---
        "540 VARIAL HEELFLIP",
        "540 VARIAL DOUBLE HEELFLIP",
        "540 VARIAL TRIPLE HEELFLIP",
        "540 VARIAL QUAD HEELFLIP",
        # --- 720 Varial heelflip series ---
        "720 VARIAL HEELFLIP",
        "720 VARIAL DOUBLE HEELFLIP",
        "720 VARIAL TRIPLE HEELFLIP",
        "720 VARIAL QUAD HEELFLIP",
        # --- Inward heelflip series (heelflip + pop shove-it) ---
        "INWARD HEELFLIP",
        "INWARD DOUBLE HEEL",
        "INWARD TRIPLE HEEL",
        "INWARD QUAD HEEL",
        # --- 360 Inward heelflip series ---
        "360 INWARD HEELFLIP",
        "360 INWARD DOUBLE HEEL",
        "360 INWARD TRIPLE HEEL",
        "360 INWARD QUAD HEEL",
        # --- 540 Inward heelflip series ---
        "540 INWARD HEELFLIP",
        "540 INWARD DOUBLE HEEL",
        "540 INWARD TRIPLE HEEL",
        "540 INWARD QUAD HEEL",
        # --- 720 Inward heelflip series ---
        "720 INWARD HEELFLIP",
        "720 INWARD DOUBLE HEEL",
        "720 INWARD TRIPLE HEEL",
        "720 INWARD QUAD HEEL",
        # --- Backside heelflip series (BS 180 + heelflip) ---
        "BACKSIDE HEEL FLIP",
        "BACKSIDE DOUBLE HEEL",
        "BACKSIDE TRIPLE HEEL",
        "BACKSIDE QUAD HEEL",
        # --- Frontside heelflip series (FS 180 + heelflip) ---
        "FRONTSIDE HEEL FLIP",
        "FRONTSIDE DOUBLE HEEL",
        "FRONTSIDE TRIPLE HEEL",
        "FRONTSIDE QUAD HEEL",
        # --- Backside 360 heel series ---
        "BACKSIDE 360 HEEL",
        "BACKSIDE 360 DOUBLE HEEL",
        "BACKSIDE 360 TRIPLE HEEL",
        "BACKSIDE 360 QUAD HEEL",
        # --- Frontside 360 heel series ---
        "FRONTSIDE 360 HEEL",
        "FRONTSIDE 360 DOUBLE HEEL",
        "FRONTSIDE 360 TRIPLE HEEL",
        "FRONTSIDE 360 QUAD HEEL",
        # --- Backside 540 heel series ---
        "BACKSIDE 540 HEEL",
        "BACKSIDE 540 DOUBLE HEEL",
        "BACKSIDE 540 TRIPLE HEEL",
        "BACKSIDE 540 QUAD HEEL",
        # --- Frontside 540 heel series ---
        "FRONTSIDE 540 HEEL",
        "FRONTSIDE 540 DOUBLE HEEL",
        "FRONTSIDE 540 TRIPLE HEEL",
        "FRONTSIDE 540 QUAD HEEL",
        # --- Backside 720 heel series ---
        "BACKSIDE 720 HEEL",
        "BACKSIDE 720 DOUBLE HEEL",
        "BACKSIDE 720 TRIPLE HEEL",
        "BACKSIDE 720 QUAD HEEL",
        # --- Frontside 720 heel series ---
        "FRONTSIDE 720 HEEL",
        "FRONTSIDE 720 DOUBLE HEEL",
        "FRONTSIDE 720 TRIPLE HEEL",
        "FRONTSIDE 720 QUAD HEEL",
    }
)

# Pop shove-it mechanic tricks.
SHOVE_IT_FAMILY: frozenset[str] = frozenset(
    {
        "FS POP SHOVE-IT",
        "POP SHOVE-IT",
        "FS 360 POP SHOVE-IT",
        "360 POP SHOVE-IT",
        "FS 540 POP SHOVE-IT",
        "540 POP SHOVE-IT",
        "FS 720 POP SHOVE-IT",
        "720 POP SHOVE-IT",
    }
)

# Pure rotation tricks: body/board rotations without flip mechanics.
ROTATION_FAMILY: frozenset[str] = frozenset(
    {
        "FRONTSIDE 180",
        "BACKSIDE 180",
        "FRONTSIDE 360",
        "BACKSIDE 360",
        "FRONTSIDE 540",
        "BACKSIDE 540",
        "FRONTSIDE 720",
        "BACKSIDE 720",
    }
)

# Grinds and slides.
GRIND_SLIDE_FAMILY: frozenset[str] = frozenset(
    {
        "50 50 GRIND",
        "5 0 GRIND",
        "BOARDSLIDE",
        "LIPSLIDE",
        "TAIL SLIDE",
        "NOSE SLIDE",
        "DARKSLIDE",
        "SUSKI GRIND",
        "SMITH GRIND",
        "FEEBLE GRIND",
        "SALAD GRIND",
        "NOSE GRIND",
        "CROOKED GRIND",
        "LAZY GRIND",
        "OVERCROOK",
        "LOSI GRIND",
        "BLUNTSLIDE",
        "NOSEBLUNT",
        "CASPER SLIDE",
        "NOSE CASPER SLIDE",
        "PRIMO SLIDE",
    }
)

# Big Spin / Bigger Spin / Gazelle Spin families (board rotation + body rotation combos).
# Includes FS/BS variants and the heel/flip sub-variants.
SPIN_FAMILY: frozenset[str] = frozenset(
    {
        # --- Standalone spin tricks ---
        # Unprefixed spin names are the backside forms in True Skate nomenclature.
        "BIG SPIN",
        "BIGGER SPIN",
        "GAZELLE SPIN",
        "FRONTSIDE BIG SPIN",
        "FRONTSIDE BIGGER SPIN",
        "FRONTSIDE GAZELLE SPIN",
        # --- Big flip ---
        "BIG FLIP",
        "BIG DOUBLE FLIP",
        "BIG TRIPLE FLIP",
        "BIG QUAD FLIP",
        # --- Big heel ---
        "BIG HEEL",
        "BIG DOUBLE HEEL",
        "BIG TRIPLE HEEL",
        "BIG QUAD HEEL",
        # --- Frontside Big flip ---
        "FRONTSIDE BIG FLIP",
        "FRONTSIDE BIG DOUBLE FLIP",
        "FRONTSIDE BIG TRIPLE FLIP",
        "FRONTSIDE BIG QUAD FLIP",
        # --- Frontside Big heel ---
        "FRONTSIDE BIG HEEL",
        "FRONTSIDE BIG DOUBLE HEEL",
        "FRONTSIDE BIG TRIPLE HEEL",
        "FRONTSIDE BIG QUAD HEEL",
        # --- Bigger flip ---
        "BIGGER FLIP",
        "BIGGER DOUBLE FLIP",
        "BIGGER TRIPLE FLIP",
        "BIGGER QUAD FLIP",
        # --- Bigger heel ---
        "BIGGER HEEL",
        "BIGGER DOUBLE HEEL",
        "BIGGER TRIPLE HEEL",
        "BIGGER QUAD HEEL",
        # --- Frontside Bigger flip ---
        "FRONTSIDE BIGGER FLIP",
        "FRONTSIDE BIGGER DOUBLE FLIP",
        "FRONTSIDE BIGGER TRIPLE FLIP",
        "FRONTSIDE BIGGER QUAD FLIP",
        # --- Frontside Bigger heel ---
        "FRONTSIDE BIGGER HEEL",
        "FRONTSIDE BIGGER DOUBLE HEEL",
        "FRONTSIDE BIGGER TRIPLE HEEL",
        "FRONTSIDE BIGGER QUAD HEEL",
        # --- Gazelle flip ---
        "GAZELLE FLIP",
        "GAZELLE DOUBLE FLIP",
        "GAZELLE TRIPLE FLIP",
        "GAZELLE QUAD FLIP",
        # --- Gazelle heel ---
        "GAZELLE HEEL",
        "GAZELLE DOUBLE HEEL",
        "GAZELLE TRIPLE HEEL",
        "GAZELLE QUAD HEEL",
        # --- Frontside Gazelle flip ---
        "FRONTSIDE GAZELLE FLIP",
        "FRONTSIDE GAZELLE DOUBLE FLIP",
        "FRONTSIDE GAZELLE TRIPLE FLIP",
        "FRONTSIDE GAZELLE QUAD FLIP",
        # --- Frontside Gazelle heel ---
        "FRONTSIDE GAZELLE HEEL",
        "FRONTSIDE GAZELLE DOUBLE HEEL",
        "FRONTSIDE GAZELLE TRIPLE HEEL",
        "FRONTSIDE GAZELLE QUAD HEEL",
    }
)

# Dolphin and Dragon trick families.
DOLPHIN_DRAGON_FAMILY: frozenset[str] = frozenset(
    {
        # --- Dolphin flip (no rotation prefix) ---
        "DOLPHIN FLIP",
        "DOLPHIN DOUBLE FLIP",
        "DOLPHIN TRIPLE FLIP",
        "DOLPHIN QUAD FLIP",
        # --- Dolphin heel (no rotation prefix) ---
        "DOLPHIN HEEL",
        "DOLPHIN DOUBLE HEEL",
        "DOLPHIN TRIPLE HEEL",
        "DOLPHIN QUAD HEEL",
        # --- 360 Dolphin flip ---
        "360 DOLPHIN FLIP",
        "360 DOLPHIN DOUBLE FLIP",
        "360 DOLPHIN TRIPLE FLIP",
        "360 DOLPHIN QUAD FLIP",
        # --- 360 Dolphin heel ---
        "360 DOLPHIN HEEL",
        "360 DOLPHIN DOUBLE HEEL",
        "360 DOLPHIN TRIPLE HEEL",
        "360 DOLPHIN QUAD HEEL",
        # --- 540 Dolphin flip ---
        "540 DOLPHIN FLIP",
        "540 DOLPHIN DOUBLE FLIP",
        "540 DOLPHIN TRIPLE FLIP",
        "540 DOLPHIN QUAD FLIP",
        # --- 540 Dolphin heel ---
        "540 DOLPHIN HEEL",
        "540 DOLPHIN DOUBLE HEEL",
        "540 DOLPHIN TRIPLE HEEL",
        "540 DOLPHIN QUAD HEEL",
        # --- 720 Dolphin flip ---
        "720 DOLPHIN FLIP",
        "720 DOLPHIN DOUBLE FLIP",
        "720 DOLPHIN TRIPLE FLIP",
        "720 DOLPHIN QUAD FLIP",
        # --- 720 Dolphin heel ---
        "720 DOLPHIN HEEL",
        "720 DOLPHIN DOUBLE HEEL",
        "720 DOLPHIN TRIPLE HEEL",
        "720 DOLPHIN QUAD HEEL",
        # --- Backside Dolphin flip ---
        "BACKSIDE DOLPHIN FLIP",
        "BACKSIDE DOLPHIN DOUBLE FLIP",
        "BACKSIDE DOLPHIN TRIPLE FLIP",
        "BACKSIDE DOLPHIN QUAD FLIP",
        # --- Backside Dolphin heel ---
        "BACKSIDE DOLPHIN HEEL",
        "BACKSIDE DOLPHIN DOUBLE HEEL",
        "BACKSIDE DOLPHIN TRIPLE HEEL",
        "BACKSIDE DOLPHIN QUAD HEEL",
        # --- Frontside Dolphin flip ---
        "FRONTSIDE DOLPHIN FLIP",
        "FRONTSIDE DOLPHIN DOUBLE FLIP",
        "FRONTSIDE DOLPHIN TRIPLE FLIP",
        "FRONTSIDE DOLPHIN QUAD FLIP",
        # --- Frontside Dolphin heel ---
        "FRONTSIDE DOLPHIN HEEL",
        "FRONTSIDE DOLPHIN DOUBLE HEEL",
        "FRONTSIDE DOLPHIN TRIPLE HEEL",
        "FRONTSIDE DOLPHIN QUAD HEEL",
        # --- Backside 360 Dolphin flip ---
        "BACKSIDE 360 DOLPHIN FLIP",
        "BACKSIDE 360 DOLPHIN DOUBLE FLIP",
        "BACKSIDE 360 DOLPHIN TRIPLE FLIP",
        "BACKSIDE 360 DOLPHIN QUAD FLIP",
        # --- Backside 360 Dolphin heel ---
        "BACKSIDE 360 DOLPHIN HEEL",
        "BACKSIDE 360 DOLPHIN DOUBLE HEEL",
        "BACKSIDE 360 DOLPHIN TRIPLE HEEL",
        "BACKSIDE 360 DOLPHIN QUAD HEEL",
        # --- Frontside 360 Dolphin flip ---
        "FRONTSIDE 360 DOLPHIN FLIP",
        "FRONTSIDE 360 DOLPHIN DOUBLE FLIP",
        "FRONTSIDE 360 DOLPHIN TRIPLE FLIP",
        "FRONTSIDE 360 DOLPHIN QUAD FLIP",
        # --- Frontside 360 Dolphin heel ---
        "FRONTSIDE 360 DOLPHIN HEEL",
        "FRONTSIDE 360 DOLPHIN DOUBLE HEEL",
        "FRONTSIDE 360 DOLPHIN TRIPLE HEEL",
        "FRONTSIDE 360 DOLPHIN QUAD HEEL",
        # --- Backside 540 Dolphin flip ---
        "BACKSIDE 540 DOLPHIN FLIP",
        "BACKSIDE 540 DOLPHIN DOUBLE FLIP",
        "BACKSIDE 540 DOLPHIN TRIPLE FLIP",
        "BACKSIDE 540 DOLPHIN QUAD FLIP",
        # --- Backside 540 Dolphin heel ---
        "BACKSIDE 540 DOLPHIN HEEL",
        "BACKSIDE 540 DOLPHIN DOUBLE HEEL",
        "BACKSIDE 540 DOLPHIN TRIPLE HEEL",
        "BACKSIDE 540 DOLPHIN QUAD HEEL",
        # --- Frontside 540 Dolphin flip ---
        "FRONTSIDE 540 DOLPHIN FLIP",
        "FRONTSIDE 540 DOLPHIN DOUBLE FLIP",
        "FRONTSIDE 540 DOLPHIN TRIPLE FLIP",
        "FRONTSIDE 540 DOLPHIN QUAD FLIP",
        # --- Frontside 540 Dolphin heel ---
        "FRONTSIDE 540 DOLPHIN HEEL",
        "FRONTSIDE 540 DOLPHIN DOUBLE HEEL",
        "FRONTSIDE 540 DOLPHIN TRIPLE HEEL",
        "FRONTSIDE 540 DOLPHIN QUAD HEEL",
        # --- Backside 720 Dolphin flip ---
        "BACKSIDE 720 DOLPHIN FLIP",
        "BACKSIDE 720 DOLPHIN DOUBLE FLIP",
        "BACKSIDE 720 DOLPHIN TRIPLE FLIP",
        "BACKSIDE 720 DOLPHIN QUAD FLIP",
        # --- Backside 720 Dolphin heel ---
        "BACKSIDE 720 DOLPHIN HEEL",
        "BACKSIDE 720 DOLPHIN DOUBLE HEEL",
        "BACKSIDE 720 DOLPHIN TRIPLE HEEL",
        "BACKSIDE 720 DOLPHIN QUAD HEEL",
        # --- Frontside 720 Dolphin flip ---
        "FRONTSIDE 720 DOLPHIN FLIP",
        "FRONTSIDE 720 DOLPHIN DOUBLE FLIP",
        "FRONTSIDE 720 DOLPHIN TRIPLE FLIP",
        "FRONTSIDE 720 DOLPHIN QUAD FLIP",
        # --- Frontside 720 Dolphin heel ---
        "FRONTSIDE 720 DOLPHIN HEEL",
        "FRONTSIDE 720 DOLPHIN DOUBLE HEEL",
        "FRONTSIDE 720 DOLPHIN TRIPLE HEEL",
        "FRONTSIDE 720 DOLPHIN QUAD HEEL",
        # --- Backside Big Dolphin flip ---
        "BACKSIDE BIG DOLPHIN FLIP",
        "BACKSIDE BIG DOLPHIN DOUBLE FLIP",
        "BACKSIDE BIG DOLPHIN TRIPLE FLIP",
        "BACKSIDE BIG DOLPHIN QUAD FLIP",
        # --- Backside Big Dolphin heel ---
        "BACKSIDE BIG DOLPHIN HEEL",
        "BACKSIDE BIG DOLPHIN DOUBLE HEEL",
        "BACKSIDE BIG DOLPHIN TRIPLE HEEL",
        "BACKSIDE BIG DOLPHIN QUAD HEEL",
        # --- Frontside Big Dolphin flip ---
        "FRONTSIDE BIG DOLPHIN FLIP",
        "FRONTSIDE BIG DOLPHIN DOUBLE FLIP",
        "FRONTSIDE BIG DOLPHIN TRIPLE FLIP",
        "FRONTSIDE BIG DOLPHIN QUAD FLIP",
        # --- Frontside Big Dolphin heel ---
        "FRONTSIDE BIG DOLPHIN HEEL",
        "FRONTSIDE BIG DOLPHIN DOUBLE HEEL",
        "FRONTSIDE BIG DOLPHIN TRIPLE HEEL",
        "FRONTSIDE BIG DOLPHIN QUAD HEEL",
        # --- Backside Bigger Dolphin flip ---
        "BACKSIDE BIGGER DOLPHIN FLIP",
        "BACKSIDE BIGGER DOLPHIN DOUBLE FLIP",
        "BACKSIDE BIGGER DOLPHIN TRIPLE FLIP",
        "BACKSIDE BIGGER DOLPHIN QUAD FLIP",
        # --- Backside Bigger Dolphin heel ---
        "BACKSIDE BIGGER DOLPHIN HEEL",
        "BACKSIDE BIGGER DOLPHIN DOUBLE HEEL",
        "BACKSIDE BIGGER DOLPHIN TRIPLE HEEL",
        "BACKSIDE BIGGER DOLPHIN QUAD HEEL",
        # --- Frontside Bigger Dolphin flip ---
        "FRONTSIDE BIGGER DOLPHIN FLIP",
        "FRONTSIDE BIGGER DOLPHIN DOUBLE FLIP",
        "FRONTSIDE BIGGER DOLPHIN TRIPLE FLIP",
        "FRONTSIDE BIGGER DOLPHIN QUAD FLIP",
        # --- Frontside Bigger Dolphin heel ---
        "FRONTSIDE BIGGER DOLPHIN HEEL",
        "FRONTSIDE BIGGER DOLPHIN DOUBLE HEEL",
        "FRONTSIDE BIGGER DOLPHIN TRIPLE HEEL",
        "FRONTSIDE BIGGER DOLPHIN QUAD HEEL",
        # --- Backside Gazelle Dolphin flip ---
        "BACKSIDE GAZELLE DOLPHIN FLIP",
        "BACKSIDE GAZELLE DOLPHIN DOUBLE FLIP",
        "BACKSIDE GAZELLE DOLPHIN TRIPLE FLIP",
        "BACKSIDE GAZELLE DOLPHIN QUAD FLIP",
        # --- Backside Gazelle Dolphin heel ---
        "BACKSIDE GAZELLE DOLPHIN HEEL",
        "BACKSIDE GAZELLE DOLPHIN DOUBLE HEEL",
        "BACKSIDE GAZELLE DOLPHIN TRIPLE HEEL",
        "BACKSIDE GAZELLE DOLPHIN QUAD HEEL",
        # --- Frontside Gazelle Dolphin flip ---
        "FRONTSIDE GAZELLE DOLPHIN FLIP",
        "FRONTSIDE GAZELLE DOLPHIN DOUBLE FLIP",
        "FRONTSIDE GAZELLE DOLPHIN TRIPLE FLIP",
        "FRONTSIDE GAZELLE DOLPHIN QUAD FLIP",
        # --- Frontside Gazelle Dolphin heel ---
        "FRONTSIDE GAZELLE DOLPHIN HEEL",
        "FRONTSIDE GAZELLE DOLPHIN DOUBLE HEEL",
        "FRONTSIDE GAZELLE DOLPHIN TRIPLE HEEL",
        "FRONTSIDE GAZELLE DOLPHIN QUAD HEEL",
        # --- Dragon flip ---
        "DRAGON FLIP",
        "BACKSIDE DRAGON FLIP",
        "FRONTSIDE DRAGON FLIP",
    }
)

# Miscellaneous tricks that don't fit a specific mechanic family.
OTHER_TRICKS: frozenset[str] = frozenset(
    {
        "OLLIE",
        "IMPOSSIBLE",
        "DOUBLE IMPOSSIBLE",
        "MANUAL",
        "NOSE MANUAL",
        "WALL RIDE",
    }
)

# ---------------------------------------------------------------------------
# Master set — broad union used for OCR fuzzy matching
# ---------------------------------------------------------------------------

KNOWN_TRICKS: frozenset[str] = (
    KICKFLIP_FAMILY
    | HEELFLIP_FAMILY
    | SHOVE_IT_FAMILY
    | ROTATION_FAMILY
    | GRIND_SLIDE_FAMILY
    | SPIN_FAMILY
    | DOLPHIN_DRAGON_FAMILY
    | OTHER_TRICKS
)

# ---------------------------------------------------------------------------
# Family lookup
# ---------------------------------------------------------------------------

_FAMILY_MAP: tuple[tuple[frozenset[str], str], ...] = (
    (KICKFLIP_FAMILY, "kickflip"),
    (HEELFLIP_FAMILY, "heelflip"),
    (SHOVE_IT_FAMILY, "shove-it"),
    (ROTATION_FAMILY, "rotation"),
    (GRIND_SLIDE_FAMILY, "grind"),
    (SPIN_FAMILY, "spin"),
    (DOLPHIN_DRAGON_FAMILY, "dolphin-dragon"),
    (OTHER_TRICKS, "other-tricks"),
)


def mechanical_family(trick: str) -> str | None:
    """Return the mechanic family name for a trick, or None if not recognised.

    Returns one of:
        'kickflip' | 'heelflip' | 'shove-it' | 'rotation' | 'grind' |
        'spin' | 'dolphin-dragon' | 'other-tricks' | None

    Lookup is by exact membership in the family frozensets (no substring
    matching), so the caller must supply the canonical trick name.
    """
    normalised = trick.strip().upper()
    for family_set, family_name in _FAMILY_MAP:
        if normalised in family_set:
            return family_name
    return None
