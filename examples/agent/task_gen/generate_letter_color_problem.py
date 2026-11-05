import os
import random
import re
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

# LETTER_BLOCKS = ["block_r"]
LETTER_BLOCKS = ["block_r"]

COLOR_BLOCKS = ["block_red1", "block_blue", "block_blue2", "block_blue3"]
# COLOR_BLOCKS = []

# Pools --num-blocks samples the scene from. Names must resolve to an asset in
# toyblocks's template/codegen.py, which spawns them: the cuboids below are its
# registered colors, and every block_a..block_z maps to the matching letter USD.
LETTER_BLOCK_POOL = [f"block_{c}" for c in "abcdefghijklmnopqrstuvwxyz"]

COLOR_BLOCK_POOL = [
    "block_red1",
    "block_red2",
    "block_red3",
    "block_red4",
    "block_red5",
    "block_blue",
    "block_blue2",
    "block_blue3",
    "block_blue4",
    "block_green",
    "block_green2",
    "block_green3",
    "block_green4",
    "block_yellow",
    "block_yellow2",
    "block_yellow3",
    "block_orange",
    "block_orange2",
    "block_purple",
    "block_purple2",
    "block_cyan",
    "block_cyan2",
    "block_gray",
    "block_gray2",
    "block_pink",
    "block_brown",
]

# Named blocks outside the pools that codegen still has an asset for.
LEGACY_BLOCKS = ["block_o1", "block_o2"]

# Glyph block names, optionally carrying a _<copy> suffix. Mirrors the pattern
# codegen resolves USDs with, so anything matching here has an asset.
_GLYPH_BLOCK_PATTERN = re.compile(r"^block_([a-z1-9])(?:_(\d+))?$")

ALL_BLOCKS = LETTER_BLOCKS + COLOR_BLOCKS

ORIENTATIONS = [f"o{i}" for i in range(1, 25)]

# Orientation that shows each block's letter facing front (readable) when stacked.
# Face layout (from Block_CharE_CharR_5 / orientation.py):
#   face one=top, two=bottom, three=front, four=back, five=left, six=right
# block_r  (Block_CharE_CharR_5): R on face five  -> o3  (up=one, front=five)
# block_o1 (block_o_b_k):         O on face five  -> o3  (up=one, front=five)
# block_b  (block_o_b_k):         B on face six   -> o4  (up=one, front=six)
# block_o2 (block_o_b_k):         O on face five  -> o3  (up=one, front=five)
# block_t  (block_t_g_x):         T on face five  -> o3  (up=one, front=five)
SPELLING_ORIENTATIONS: dict[str, list[str]] = {
    "block_r": ["o2"],
    "block_t": ["o4"],
    "block_o1": ["o4", "o12"],
    "block_b": ["o10"],
    "block_o2": ["o4", "o12"],
}

FLIP_FORWARD = {
    "o1": "o5",
    "o2": "o17",
    "o3": "o15",
    "o4": "o23",
    "o5": "o9",
    "o6": "o18",
    "o7": "o3",
    "o8": "o22",
    "o9": "o13",
    "o10": "o19",
    "o11": "o7",
    "o12": "o21",
    "o13": "o1",
    "o14": "o20",
    "o15": "o11",
    "o16": "o24",
    "o17": "o12",
    "o18": "o16",
    "o19": "o4",
    "o20": "o8",
    "o21": "o2",
    "o22": "o14",
    "o23": "o10",
    "o24": "o6",
}
FLIP_BACKWARD = {
    "o1": "o13",
    "o2": "o21",
    "o3": "o7",
    "o4": "o19",
    "o5": "o1",
    "o6": "o24",
    "o7": "o11",
    "o8": "o20",
    "o9": "o5",
    "o10": "o23",
    "o11": "o15",
    "o12": "o17",
    "o13": "o9",
    "o14": "o22",
    "o15": "o3",
    "o16": "o18",
    "o17": "o2",
    "o18": "o6",
    "o19": "o10",
    "o20": "o14",
    "o21": "o12",
    "o22": "o8",
    "o23": "o4",
    "o24": "o16",
}
FLIP_LEFT = {
    "o1": "o20",
    "o2": "o14",
    "o3": "o22",
    "o4": "o8",
    "o5": "o17",
    "o6": "o2",
    "o7": "o21",
    "o8": "o12",
    "o9": "o18",
    "o10": "o6",
    "o11": "o24",
    "o12": "o16",
    "o13": "o19",
    "o14": "o10",
    "o15": "o23",
    "o16": "o4",
    "o17": "o15",
    "o18": "o3",
    "o19": "o7",
    "o20": "o11",
    "o21": "o13",
    "o22": "o9",
    "o23": "o5",
    "o24": "o1",
}
FLIP_RIGHT = {
    "o1": "o24",
    "o2": "o6",
    "o3": "o18",
    "o4": "o16",
    "o5": "o23",
    "o6": "o10",
    "o7": "o19",
    "o8": "o4",
    "o9": "o22",
    "o10": "o14",
    "o11": "o20",
    "o12": "o8",
    "o13": "o21",
    "o14": "o2",
    "o15": "o17",
    "o16": "o12",
    "o17": "o5",
    "o18": "o9",
    "o19": "o13",
    "o20": "o1",
    "o21": "o7",
    "o22": "o3",
    "o23": "o15",
    "o24": "o11",
}
FLIP_TABLES = {
    "forward": FLIP_FORWARD,
    "backward": FLIP_BACKWARD,
    "left_flip": FLIP_LEFT,
    "right_flip": FLIP_RIGHT,
}

YAW_CW = {
    "o1": "o2",
    "o2": "o3",
    "o3": "o4",
    "o4": "o1",
    "o5": "o6",
    "o6": "o7",
    "o7": "o8",
    "o8": "o5",
    "o9": "o10",
    "o10": "o11",
    "o11": "o12",
    "o12": "o9",
    "o13": "o14",
    "o14": "o15",
    "o15": "o16",
    "o16": "o13",
    "o17": "o18",
    "o18": "o19",
    "o19": "o20",
    "o20": "o17",
    "o21": "o22",
    "o22": "o23",
    "o23": "o24",
    "o24": "o21",
}
YAW_CCW = {v: k for k, v in YAW_CW.items()}

LETTER_BLOCK_TYPES = {
    "r": "block_r_e",
    "o": "block_o_b",
    "t": "block_t_g",
    "b": "block_o_b",
}

# Two-instance same-skill goal modes. Each name maps to (kind, direction); the
# generator emits exactly two independent goal predicates of that skill on
# distinct blocks, with every participating block forced on_table + clear in the
# init (the mover/flipper/rotator always starts on the table). flip/rotate need
# two mover blocks; align_right/pnp need two (mover, target) pairs.
PAIR_SKILLS: dict[str, tuple[str, str | None]] = {
    "flip_forward": ("flip", "forward"),
    "flip_backward": ("flip", "backward"),
    "flip_left": ("flip", "left_flip"),
    "flip_right": ("flip", "right_flip"),
    "rotate_cw": ("rotate", "cw"),
    "rotate_ccw": ("rotate", "ccw"),
    "align_right": ("align_right", None),
    "pnp": ("pnp", None),
}

# Blocks that must be on_table + clear for a given pair-skill kind.
PAIR_SKILL_ACTIVE_COUNT: dict[str, int] = {
    "flip": 2,
    "rotate": 2,
    "align_right": 4,
    "pnp": 4,
}

# Relative likelihood of each predicate kind in a --random goal. Orientation and
# lateral placement are weighted up because they exercise reorient and
# push_to_right, which stacking goals never reach.
GOAL_PREDICATE_WEIGHTS: dict[str, float] = {
    "ori_of": 3.0,
    "on_right": 3.0,
    "on": 1.0,
    "on_table": 1.0,
}


def parse_goal_weights(spec: str) -> dict[str, float]:
    """Parse ``ori_of=3,on_right=3,on=1,on_table=1`` into a weight dict.

    Unlisted kinds keep their default weight; a weight of 0 drops that kind.
    """
    weights = dict(GOAL_PREDICATE_WEIGHTS)
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        kind, _, raw = item.partition("=")
        kind = kind.strip()
        if kind not in GOAL_PREDICATE_WEIGHTS:
            raise ValueError(
                f"Unknown goal predicate {kind!r}. Choose from {sorted(GOAL_PREDICATE_WEIGHTS)}."
            )
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"Weight for {kind!r} must be a number, got {raw!r}") from None
        if value < 0:
            raise ValueError(f"Weight for {kind!r} must be non-negative, got {value}")
        weights[kind] = value
    if not any(weights.values()):
        raise ValueError("At least one goal predicate must have a non-zero weight")
    return weights


def _block_type(block_name: str) -> str:
    """Return the PDDL type for a block based on LETTER_BLOCK_TYPES.

    Letters outside LETTER_BLOCK_TYPES fall back to ``block_letter``, the generic
    glyph type toyblocks's domain.pddl declares, so a sampled ``block_k`` still
    reads as a letter rather than an anonymous ``block``.
    """
    if block_name in LETTER_BLOCKS:
        letter = block_to_letter(block_name).lower()
        if letter in LETTER_BLOCK_TYPES:
            return LETTER_BLOCK_TYPES[letter]
        return "block_letter" if len(letter) == 1 and letter.isalpha() else "block"
    return "block"


def _format_objects() -> str:
    """Format the :objects section, grouping blocks by their PDDL type."""
    type_groups: dict[str, list[str]] = {}
    for b in ALL_BLOCKS:
        t = _block_type(b)
        type_groups.setdefault(t, []).append(b)

    lines = []
    for t, blocks in type_groups.items():
        lines.append(f"    {' '.join(blocks)} - {t}")
    lines.append("    workspace_table - table")
    lines.append("    robot - gripper")
    return "\n".join(lines)


def _validate_blocks(block_names, param_name: str):
    """Validate that all block names exist in ALL_BLOCKS."""
    invalid = set(block_names) - set(ALL_BLOCKS)
    if invalid:
        raise ValueError(
            f"Unknown blocks in {param_name}: {sorted(invalid)}. Valid blocks: {ALL_BLOCKS}"
        )


def block_to_letter(block_name: str) -> str:
    """Extract the letter from a letter block name. e.g. block_o1 -> O, block_r_2 -> R"""
    suffix = block_name.replace("block_", "")
    return re.sub(r"_?\d+$", "", suffix).upper()


def glyph_copy_name(base: str, copy_idx: int) -> str:
    """Object name for the ``copy_idx``-th copy of a glyph block (1-based).

    Matches toyblocks's ``glyph_blocks.block_name``, whose ``_<copy>`` suffix
    codegen strips when resolving the USD, so every copy spawns the same asset.
    """
    return base if copy_idx == 1 else f"{base}_{copy_idx}"


def expand_block_specs(specs: list[str]) -> list[str]:
    """Expand ``NAME`` / ``NAME:COUNT`` specs into distinct object names.

    Counts above one only work for glyph blocks (``block_a`` .. ``block_z``,
    ``block_1`` .. ``block_9``), which have the ``_<copy>`` naming convention:
    ``block_r:3`` becomes ``block_r block_r_2 block_r_3``. Color cuboids are
    individually named assets with no copy convention, so ask for the ones you
    want by name instead.
    """
    names: list[str] = []
    for spec in specs:
        name, _, raw_count = spec.partition(":")
        is_glyph = _GLYPH_BLOCK_PATTERN.match(name) is not None
        if not is_glyph and name not in COLOR_BLOCK_POOL and name not in LEGACY_BLOCKS:
            raise ValueError(
                f"Unknown block {name!r}. Use a glyph block (block_a..block_z, "
                f"block_1..block_9), a color cube from COLOR_BLOCK_POOL, or one of "
                f"{LEGACY_BLOCKS}."
            )
        if not raw_count:
            names.append(name)
            continue
        if not raw_count.isdigit() or int(raw_count) < 1:
            raise ValueError(f"Count in {spec!r} must be a positive integer")
        count = int(raw_count)
        if count > 1 and not is_glyph:
            raise ValueError(
                f"{name!r} is a color cube, which has no copy convention, so "
                f"{spec!r} is not expressible. Name {count} distinct cubes instead."
            )
        names.extend(glyph_copy_name(name, i + 1) for i in range(count))

    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"Block(s) {duplicates} requested more than once")
    return names


def validate_spelling(spelling: str) -> list[str]:
    """
    Validate that LETTER_BLOCKS can spell the given word.
    Returns an ordered list of block names, one per character position.
    """
    available: dict[str, list[str]] = {}
    for b in LETTER_BLOCKS:
        letter = block_to_letter(b)
        available.setdefault(letter, []).append(b)

    required = Counter(spelling.upper())
    for letter, count in required.items():
        if letter not in available:
            raise ValueError(
                f"Cannot spell '{spelling}': no block for letter '{letter}'. "
                f"Available letters: {sorted(available.keys())}"
            )
        if len(available[letter]) < count:
            raise ValueError(
                f"Cannot spell '{spelling}': need {count} block(s) for '{letter}' "
                f"but only have {len(available[letter])}. Available: {available[letter]}"
            )

    used: dict[str, int] = dict.fromkeys(available, 0)
    assignment = []
    for ch in spelling.upper():
        assignment.append(available[ch][used[ch]])
        used[ch] += 1
    return assignment


def generate_random_stacking(
    blocks: list[str],
    must_be_clear: set[str] = None,
    must_on_table: set[str] = None,
) -> tuple[dict, dict]:
    """
    Randomly assign each block to be on_table or on another block.
    Blocks in must_on_table are forced onto the table.
    Blocks in must_be_clear are guaranteed to have nothing on top.
    """
    if must_be_clear is None:
        must_be_clear = set()
    if must_on_table is None:
        must_on_table = set()

    table = "table"
    on_rel = {}
    above = {b: [] for b in blocks}

    shuffled = blocks.copy()
    random.shuffle(shuffled)

    for b in shuffled:
        if b in must_on_table:
            on_rel[b] = table
            continue
        if random.random() < 0.4:
            possible_below = [
                x
                for x in blocks
                if x != b
                and x in on_rel  # already placed
                and len(above[x]) == 0  # nothing on top yet
                and x not in must_be_clear  # not forced clear
            ]
            if possible_below:
                below = random.choice(possible_below)
                on_rel[b] = below
                above[below].append(b)
                continue
        on_rel[b] = table

    return on_rel, above


def generate_init(
    init_on_table: set[str] = None,
    init_clear: set[str] = None,
    init_orientations: dict[str, str] = None,
) -> tuple[str, dict[str, str]]:
    """Returns (init_text, orientations_dict_for_all_blocks)."""
    if init_on_table is None:
        init_on_table = set()
    if init_clear is None:
        init_clear = set()
    if init_orientations is None:
        init_orientations = {}

    on_rel, above = generate_random_stacking(ALL_BLOCKS, init_clear, init_on_table)

    init = []
    for b in ALL_BLOCKS:
        if on_rel[b] == "table":
            init.append(f"        (on_table {b} workspace_table)")
        else:
            init.append(f"        (on {b} {on_rel[b]})")

    for b in ALL_BLOCKS:
        if not above[b]:
            init.append(f"        (clear {b})")

    init.append("        (handempty robot)")

    init.append("")
    all_oris = {}
    for b in ALL_BLOCKS:
        ori = init_orientations.get(b, random.choice(ORIENTATIONS))
        init.append(f"        (ori_of {b} {ori})")
        all_oris[b] = ori

    init.append("")
    init.append("        (gripper_at_init)")

    return "\n".join(init), all_oris


def generate_spelling_goal(
    spelling: str,
    goal_orientations: dict[str, str] = None,
    init_orientations: dict[str, str] = None,
) -> str:
    """Generate an existentially-quantified goal for spelling a word (top to bottom).

    Uses typed variables (``exists``) so the planner can choose any block of
    the correct type for each position.  Inequality constraints prevent the
    same block from filling two slots.  Orientation constraints ensure the
    right letter faces front.
    """
    blocks = validate_spelling(spelling)

    merged_orientations: dict[str, list[str]] = {
        b: list(SPELLING_ORIENTATIONS[b]) for b in blocks if b in SPELLING_ORIENTATIONS
    }
    if goal_orientations:
        for b, ori in goal_orientations.items():
            merged_orientations[b] = [ori]

    var_names = [f"?b{i + 1}" for i in range(len(blocks))]
    var_types = [_block_type(b) for b in blocks]

    var_decls = " ".join(f"{v} - {t}" for v, t in zip(var_names, var_types))

    inner = []

    for i in range(len(var_names)):
        for j in range(i + 1, len(var_names)):
            if var_types[i] == var_types[j]:
                inner.append(f"                (not (= {var_names[i]} {var_names[j]}))")

    for i in range(len(blocks) - 1):
        inner.append(f"                (on {var_names[i]} {var_names[i + 1]})")
    inner.append(f"                (on_table {var_names[-1]} workspace_table)")

    for i, b in enumerate(blocks):
        if b in merged_orientations:
            oris = merged_orientations[b]
            if len(oris) == 1:
                inner.append(f"                (ori_of {var_names[i]} {oris[0]})")
            else:
                or_clauses = " ".join(f"(ori_of {var_names[i]} {o})" for o in oris)
                inner.append(f"                (or {or_clauses})")

    inner_text = "\n".join(inner)

    return (
        f"\n        (exists ({var_decls})\n"
        f"            (and \n"
        f"{inner_text}\n"
        f"            )\n"
        f"        )"
    )


def generate_random_goal(
    goal_letter: str,
    goal_color: str,
    init_orientations: dict[str, str],
    goal_orientations: dict[str, str] = None,
) -> str:
    """Goal: place goal_letter on goal_color with spelling orientation."""
    if goal_orientations is None:
        goal_orientations = {}

    goal_parts = [f"        (on {goal_letter} {goal_color})"]

    if goal_letter in goal_orientations:
        goal_parts.append(f"        (ori_of {goal_letter} {goal_orientations[goal_letter]})")
    elif goal_letter in SPELLING_ORIENTATIONS:
        oris = SPELLING_ORIENTATIONS[goal_letter]
        if len(oris) == 1:
            goal_parts.append(f"        (ori_of {goal_letter} {oris[0]})")
        else:
            or_clauses = " ".join(f"(ori_of {goal_letter} {o})" for o in oris)
            goal_parts.append(f"        (or {or_clauses})")

    return "\n".join(goal_parts)


def _format_goal_condition(text: str) -> str:
    """Ensure proper 8-space indentation for goal condition text."""
    text = text.strip()
    if "\n" not in text and text.count("(") > 1:
        parts = re.findall(r"\([^)]+\)", text)
    else:
        parts = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(f"        {p}" for p in parts)


def generate_fully_random_goal(
    blocks: list[str],
    num_goal_predicates: int | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Generate a random goal, drawing predicate kinds by weight.

    Candidates are built per kind and then drawn one at a time with ``weights``,
    so the goal mix is controlled directly rather than falling out of how many
    candidates each kind happens to have. Three consistency rules are enforced
    while drawing:

    * one placement per block — ``on``, ``on_table`` and ``on_right`` compete for
      the same slot, so a block is never told to be in two places;
    * one ``ori_of`` per block, and only for letter blocks, since a colour cuboid
      has no distinguishable orientation;
    * ``on_right`` pairs stay independent — no block appears in two of them. Chains
      are what collapse in execution (each push disturbs blocks placed earlier and
      every ``on_right`` must hold at once), so they are never built.

    ``num_goal_predicates`` caps the goal size; it defaults to one placement per
    block plus one orientation per letter block.
    """
    weights = weights or GOAL_PREDICATE_WEIGHTS
    on_rel, _above = generate_random_stacking(blocks)

    table_blocks = [b for b in blocks if on_rel[b] == "table"]
    letters = [b for b in blocks if b not in COLOR_BLOCKS]
    # Each entry is (consistency-key, predicate text).
    candidates: dict[str, list[tuple[object, str]]] = {
        "on": [(b, f"(on {b} {on_rel[b]})") for b in blocks if on_rel[b] != "table"],
        "on_table": [(b, f"(on_table {b} workspace_table)") for b in table_blocks],
        "ori_of": [(b, f"(ori_of {b} {random.choice(ORIENTATIONS)})") for b in letters],
        # on_right needs equal z, so both blocks must be on the table in the goal.
        "on_right": [
            ((a, b), f"(on_right {a} {b})") for a in table_blocks for b in table_blocks if a != b
        ],
    }
    for entries in candidates.values():
        random.shuffle(entries)

    target = num_goal_predicates
    if target is None:
        target = len(blocks) + len(letters)

    placed: set[str] = set()
    oriented: set[str] = set()
    paired: set[str] = set()
    goal_parts: list[str] = []
    while len(goal_parts) < target:
        kinds = [k for k, v in candidates.items() if v and weights.get(k, 0) > 0]
        if not kinds:
            break
        kind = random.choices(kinds, weights=[weights[k] for k in kinds])[0]
        key, text = candidates[kind].pop()
        if kind == "on_right":
            a, b = key  # type: ignore[misc]
            if a in placed or a in paired or b in paired:
                continue
            placed.add(a)
            paired.update((a, b))
        elif kind == "ori_of":
            if key in oriented:
                continue
            oriented.add(key)  # type: ignore[arg-type]
        else:
            if key in placed:
                continue
            placed.add(key)  # type: ignore[arg-type]
        goal_parts.append(f"        {text}")

    stacking_desc = ", ".join(p.strip() for p in goal_parts)
    goal_desc = f"random goal [{len(goal_parts)} predicates] ({stacking_desc})"

    return "\n".join(goal_parts), goal_desc


def generate_flip_goal(
    blocks: list[str],
    init_orientations: dict[str, str],
) -> tuple[str, str]:
    """Pick a random block, flip it in a random direction, then place it on a random other block."""
    block = random.choice(blocks)
    direction = random.choice(list(FLIP_TABLES.keys()))
    table = FLIP_TABLES[direction]
    init_ori = init_orientations[block]
    goal_ori = table[init_ori]
    target = random.choice([b for b in blocks if b != block])
    goal_parts = [
        f"        (ori_of {block} {goal_ori})",
        f"        (on {block} {target})",
    ]
    goal_desc = f"flip {block} {direction}: {init_ori} -> {goal_ori}, then on {target}"
    return "\n".join(goal_parts), goal_desc


def generate_rotate_goal(
    blocks: list[str],
    init_orientations: dict[str, str],
    include_placement: bool = True,
    all_blocks: list[str] = None,
) -> tuple[str, str]:
    """Pick a random block, rotate it CW or CCW, optionally place on another block."""
    block = random.choice(blocks)
    direction = random.choice(["cw", "ccw"])
    table = YAW_CW if direction == "cw" else YAW_CCW
    init_ori = init_orientations[block]
    goal_ori = table[init_ori]
    goal_parts = [f"        (ori_of {block} {goal_ori})"]
    goal_desc = f"rotate {block} {direction}: {init_ori} -> {goal_ori}"
    if include_placement:
        pool = all_blocks if all_blocks else blocks
        target = random.choice([b for b in pool if b != block])
        goal_parts.append(f"        (on {block} {target})")
        goal_desc += f", then on {target}"
    return "\n".join(goal_parts), goal_desc


def generate_two_instance_goal(
    active_blocks: list[str],
    kind: str,
    direction: str | None,
    init_orientations: dict[str, str],
) -> tuple[str, str]:
    """Two independent instances of the same skill on distinct blocks.

    flip/rotate: ``active_blocks`` are two movers; each is reoriented one step in
    ``direction`` (goal is orientation only, block stays on the table).
    align_right/pnp: ``active_blocks`` are ``[mover1, target1, mover2, target2]``;
    the goal is one ``on_right`` / ``on`` predicate per pair. The two pairs are
    independent (no shared block), matching how ``generate_fully_random_goal``
    keeps ``on_right`` pairs from chaining.
    """
    if kind in ("flip", "rotate"):
        if kind == "flip":
            table = FLIP_TABLES[direction]
        else:
            table = YAW_CW if direction == "cw" else YAW_CCW
        goal_parts: list[str] = []
        descs: list[str] = []
        for mover in active_blocks:
            init_ori = init_orientations[mover]
            goal_ori = table[init_ori]
            goal_parts.append(f"        (ori_of {mover} {goal_ori})")
            descs.append(f"{mover} {init_ori}->{goal_ori}")
        return "\n".join(goal_parts), f"{kind} {direction} x2 ({'; '.join(descs)})"
    if kind == "align_right":
        m1, t1, m2, t2 = active_blocks
        goal_parts = [
            f"        (on_right {m1} {t1})",
            f"        (on_right {m2} {t2})",
        ]
        return "\n".join(goal_parts), f"align_right x2 ({m1}->{t1}; {m2}->{t2})"
    if kind == "pnp":
        top1, bot1, top2, bot2 = active_blocks
        goal_parts = [
            f"        (on {top1} {bot1})",
            f"        (on {top2} {bot2})",
        ]
        return "\n".join(goal_parts), f"pnp x2 (on {top1} {bot1}; on {top2} {bot2})"
    raise ValueError(f"unknown pair-skill kind {kind!r}")


def _generate_content(
    init_on_table: set[str] = None,
    init_clear: set[str] = None,
    init_orientations: dict[str, str] = None,
    goal_spelling: str = None,
    goal_condition: str = None,
    goal_orientations: dict[str, str] = None,
    hard: bool = False,
    fully_random: bool = False,
    num_goal_predicates: int | None = None,
    goal_weights: dict[str, float] | None = None,
    rotate: bool = False,
    flip: bool = False,
    rotate_no_place: bool = False,
    easy: bool = False,
    two_skill: str | None = None,
) -> tuple[str, str, str]:
    """
    Generate init and goal PDDL content.
    Returns (init_text, goal_text, goal_description).
    """
    if goal_spelling and goal_condition:
        raise ValueError("Cannot specify both --goal-spelling and --goal-condition")

    goal_letter = goal_color = None

    # Pre-select the blocks that take part in a two-instance skill goal so the
    # init can force them on_table + clear before orientations are assigned.
    pair_kind = pair_dir = None
    active_pair: list[str] | None = None
    if two_skill:
        pair_kind, pair_dir = PAIR_SKILLS[two_skill]
        n_active = PAIR_SKILL_ACTIVE_COUNT[pair_kind]
        if len(ALL_BLOCKS) < n_active:
            raise ValueError(
                f"--two-skill {two_skill} needs {n_active} distinct blocks but the "
                f"scene has {len(ALL_BLOCKS)}; raise --num-blocks."
            )
        active_pair = random.sample(ALL_BLOCKS, n_active)
        init_on_table = set(active_pair)
        init_clear = set(active_pair)
    elif rotate or flip:
        init_on_table = set(ALL_BLOCKS)
        init_clear = set(ALL_BLOCKS)
    elif fully_random:
        pass
    elif goal_spelling:
        validate_spelling(goal_spelling)
    elif goal_condition is not None:
        if easy:
            # Force the pick block on_table + clear so it is graspable, but only
            # require the place block to be clear (nothing on its top). The place
            # block is deliberately *not* pinned to the table, so random stacking
            # may bury it on a distractor while still leaving it clear.
            _on_re = re.compile(r"\(on\s+(\w+)\s+(\w+)\)")
            for m in _on_re.finditer(goal_condition):
                pick_b, place_b = m.group(1), m.group(2)
                init_on_table = (init_on_table or set()) | {pick_b}
                init_clear = (init_clear or set()) | {pick_b, place_b}
    elif goal_condition is None:
        if LETTER_BLOCKS and COLOR_BLOCKS:
            goal_letter = random.choice(LETTER_BLOCKS)
            goal_color = random.choice(COLOR_BLOCKS)
        elif not LETTER_BLOCKS and not COLOR_BLOCKS:
            raise ValueError(
                "Both LETTER_BLOCKS and COLOR_BLOCKS are empty, cannot generate a goal"
            )
        else:
            fully_random = True
        if not hard and goal_letter and goal_color:
            init_clear = (init_clear or set()) | {goal_letter, goal_color}

    init_text, all_oris = generate_init(
        init_on_table=init_on_table,
        init_clear=init_clear,
        init_orientations=init_orientations,
    )

    if two_skill:
        goal_text, goal_desc = generate_two_instance_goal(
            active_pair, pair_kind, pair_dir, all_oris
        )
    elif flip:
        goal_text, goal_desc = generate_flip_goal(ALL_BLOCKS, all_oris)
    elif rotate:
        rotate_src = [b for b in LETTER_BLOCKS if b in ALL_BLOCKS] or ALL_BLOCKS
        goal_text, goal_desc = generate_rotate_goal(
            rotate_src,
            all_oris,
            include_placement=not rotate_no_place,
            all_blocks=ALL_BLOCKS,
        )
    elif fully_random:
        goal_text, goal_desc = generate_fully_random_goal(
            ALL_BLOCKS, num_goal_predicates, goal_weights
        )
    elif goal_condition is not None:
        goal_text = _format_goal_condition(goal_condition)
        goal_desc = "custom goal condition"
    elif goal_spelling is not None:
        goal_text = generate_spelling_goal(goal_spelling, goal_orientations, all_oris)
        goal_desc = f"spell '{goal_spelling.upper()}'"
    else:
        goal_text = generate_random_goal(goal_letter, goal_color, all_oris, goal_orientations)
        goal_desc = f"(on {goal_letter} {goal_color})"

    return init_text, goal_text, goal_desc


def generate_problem(
    out_dir=None,
    seed=None,
    seen=None,
    max_retries=1000,
    verbose=True,
    init_on_table: set[str] = None,
    init_clear: set[str] = None,
    init_orientations: dict[str, str] = None,
    goal_spelling: str = None,
    goal_condition: str = None,
    goal_orientations: dict[str, str] = None,
    single_pick_place: bool = False,
    pick_block: str = None,
    place_block: str = None,
    hard: bool = False,
    fully_random: bool = False,
    num_goal_predicates: int | None = None,
    goal_weights: dict[str, float] | None = None,
    rotate: bool = False,
    flip: bool = False,
    rotate_no_place: bool = False,
    easy: bool = False,
    two_skill: str | None = None,
    block_size_range: tuple[float, float] | None = None,
    table_height_range: tuple[float, float] | None = None,
    friction_range: tuple[float, float] | None = None,
):
    """Generate a unique problem. seen is a set of (init_text, goal_text) fingerprints."""
    if seed is not None:
        random.seed(seed)

    if seen is None:
        seen = set()

    if rotate and flip:
        raise ValueError("--rotate and --flip are mutually exclusive")

    if two_skill:
        if rotate or flip or fully_random or goal_spelling or goal_condition or single_pick_place:
            raise ValueError(
                "--two-skill cannot be combined with --rotate/--flip/--random/"
                "--goal-spelling/--goal-condition/--single-pick-place"
            )

    if rotate:
        if goal_spelling or goal_condition or single_pick_place or fully_random:
            raise ValueError("--rotate cannot be combined with other goal modes")

    if flip:
        if goal_spelling or goal_condition or single_pick_place or fully_random:
            raise ValueError("--flip cannot be combined with other goal modes")

    if fully_random:
        if goal_spelling or goal_condition or single_pick_place:
            raise ValueError(
                "--random cannot be combined with --goal-spelling, --goal-condition, or --single-pick-place"
            )

    if single_pick_place:
        if goal_spelling or goal_condition:
            raise ValueError(
                "--single-pick-place cannot be combined with --goal-spelling or --goal-condition"
            )
        pb = pick_block or random.choice(ALL_BLOCKS)
        plb = place_block or random.choice([b for b in ALL_BLOCKS if b != pb])
        init_on_table = (init_on_table or set()) | {pb, plb}
        init_clear = (init_clear or set()) | {pb, plb}
        goal_condition = f"(on {pb} {plb})"

    for attempt in range(max_retries):
        init_text, goal_text, goal_desc = _generate_content(
            init_on_table=init_on_table,
            init_clear=init_clear,
            init_orientations=init_orientations,
            goal_spelling=goal_spelling,
            goal_condition=goal_condition,
            goal_orientations=goal_orientations,
            hard=hard,
            fully_random=fully_random,
            num_goal_predicates=num_goal_predicates,
            goal_weights=goal_weights,
            rotate=rotate,
            flip=flip,
            rotate_no_place=rotate_no_place,
            easy=easy,
            two_skill=two_skill,
        )
        fingerprint = (init_text, goal_text)
        if fingerprint not in seen:
            seen.add(fingerprint)
            break
    else:
        print(
            f"Warning: could not generate a unique problem after {max_retries} attempts, writing duplicate"
        )

    objects = _format_objects()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    random_suffix = random.randint(0, 9999)
    problem_name = f"letter-color-10-{timestamp}_{random_suffix:04d}"

    problem_pddl = (
        f"(define (problem {problem_name})\n"
        f"    (:domain blocks-tamp)\n"
        f"    (:objects\n"
        f"{objects}    )\n"
        f"    (:init\n"
        f"{init_text}\n"
        f"    )\n"
        f"    (:goal (and\n"
        f"{goal_text}\n"
        f"    ))\n"
        f")"
    )

    # toyblocks's codegen reads the scene randomization off these comments
    # (parse_pddl_randomization_ranges_from_text); without them it falls back to
    # fixed 0.04 blocks and default friction. Same placement and spelling as
    # ProblemSpec._build_pddl so the one parser handles both.
    range_comments = [
        f"    ;{name} = {list(rng)}"
        for name, rng in (
            ("block_size_range", block_size_range),
            ("table_height_range", table_height_range),
            ("friction_range", friction_range),
        )
        if rng is not None
    ]
    if range_comments:
        problem_pddl = problem_pddl[:-1] + "\n".join([""] + range_comments) + "\n)"

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(__file__), "Tasks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"problem_{problem_name}.pddl")

    with open(out_path, "w") as f:
        f.write(problem_pddl)
    if verbose:
        print(f"Generated: {out_path}")
        print(f"  Goal: {goal_desc}")
    return out_path


def _parse_key_value_pairs(pairs: list[str]) -> dict[str, str]:
    """Parse 'key=value' pairs into a dict."""
    result = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected 'block=value' format, got '{pair}'")
        k, v = pair.split("=", 1)
        result[k] = v
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a random letter-on-color blocks PDDL problem with 10 blocks."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: Tasks/ next to this script)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--count", type=int, default=1, help="Number of problems to generate")
    parser.add_argument(
        "--init-on-table",
        nargs="*",
        default=None,
        help="Blocks that must start on the table (e.g. block_r block_t)",
    )
    parser.add_argument(
        "--init-clear",
        nargs="*",
        default=None,
        help="Blocks that must start clear (e.g. block_r block_o1)",
    )
    parser.add_argument(
        "--init-ori",
        nargs="*",
        default=None,
        help="Initial orientations as block=ori (e.g. block_r=o1 block_t=o3)",
    )
    parser.add_argument(
        "--goal-spelling",
        type=str,
        default=None,
        help="Word to spell with letter blocks (e.g. ROBOT). "
        "Validates that all required letters are available.",
    )
    parser.add_argument(
        "--goal-condition",
        type=str,
        nargs="+",
        default=None,
        help="Explicit goal condition PDDL text. "
        "Multiple values are treated as alternatives (OR): "
        "each problem randomly picks one. "
        '(e.g. "(on block_r block_red1)" "(on block_red1 block_r)")',
    )
    parser.add_argument(
        "--goal-ori",
        nargs="*",
        default=None,
        help="Goal orientations as block=ori (e.g. block_r=o1 block_t=o3)",
    )
    parser.add_argument(
        "--single-pick-place",
        action="store_true",
        help="Generate a single pick-and-place problem. Both blocks start on table and clear.",
    )
    parser.add_argument(
        "--pick-block",
        type=str,
        default=None,
        help="Block to pick up (only with --single-pick-place). Random if not specified.",
    )
    parser.add_argument(
        "--place-block",
        type=str,
        default=None,
        help="Block to place on (only with --single-pick-place). Random if not specified.",
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Do not automatically clear the goal blocks in the "
        "initial state. The planner may need to unstack "
        "blocks before achieving the goal.",
    )
    parser.add_argument(
        "--easy",
        action="store_true",
        help="For --goal-condition problems, force the blocks "
        "named in each (on X Y) goal to start on_table and "
        "clear. Without this flag no block is forced onto "
        "the table and the initial stacking is fully random.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Completely random initial and goal states. "
        "Both stacking and orientations are randomized "
        "with no spelling or letter-on-color constraints.",
    )
    parser.add_argument(
        "--num-goal-predicates",
        type=int,
        default=None,
        help="Number of goal predicates to keep (only with --random). "
        "Defaults to all (2 * num_blocks). Use a smaller value "
        "to reduce problem complexity.",
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotation-only task: all blocks on table and clear, "
        "goal is to rotate a random block CW or CCW by one step.",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip-only task: all blocks on table and clear, "
        "goal is to flip a random block in a random direction "
        "(forward, backward, left_flip, right_flip).",
    )
    parser.add_argument(
        "--rotate-no-place",
        action="store_true",
        help="With --rotate: goal is orientation only, no placement.",
    )
    parser.add_argument(
        "--two-skill",
        type=str,
        default=None,
        choices=sorted(PAIR_SKILLS.keys()),
        help="Two-instance same-skill task: emit exactly two independent "
        "goal predicates of the chosen skill on distinct blocks, with "
        "every participating block forced on_table + clear in the init. "
        "flip_*/rotate_* reorient two movers one step in that direction; "
        "align_right/pnp place two (mover, target) pairs. Combine with "
        "--num-blocks to size the scene (e.g. 5).",
    )
    parser.add_argument(
        "--letter-only",
        action="store_true",
        help="Use only letter blocks (COLOR_BLOCKS becomes empty).",
    )
    parser.add_argument(
        "--color-only",
        action="store_true",
        help="Use only color blocks (LETTER_BLOCKS becomes empty).",
    )
    parser.add_argument(
        "--letter-blocks",
        nargs="*",
        default=None,
        help="Override LETTER_BLOCKS list. Pass with no args for empty list.",
    )
    parser.add_argument(
        "--color-blocks",
        nargs="*",
        default=None,
        help="Override COLOR_BLOCKS list. Pass with no args for empty list.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Total blocks in the scene, sampled per problem from the letter and "
        "colour pools (all 26 letters plus every registered cube, narrowed "
        "by --letter-blocks / --color-blocks / --letter-only / --color-only). "
        "Every block takes part in the goal; this script has no distractors.",
    )
    parser.add_argument(
        "--goal-weights",
        default=None,
        metavar="KIND=W,...",
        help="Relative likelihood of each predicate kind in a --random goal, e.g. "
        '"ori_of=3,on_right=3,on=1,on_table=1" (the default). Unlisted kinds '
        "keep their default weight; 0 drops a kind. on_right pairs are always "
        "independent, never chained.",
    )
    parser.add_argument(
        "--require-blocks",
        nargs="*",
        default=None,
        metavar="NAME[:COUNT]",
        help='Blocks that must appear in every sampled scene, e.g. "block_r" or '
        '"block_r:3 block_o:2 block_t:5". --num-blocks then fills the '
        "remainder at random; omit it to use exactly these blocks. Counts "
        "above one need a glyph block, which gets _<copy>-suffixed names.",
    )
    # The three range flags below exist so this script satisfies the same
    # task-gen contract as scripts/task_gen/*.py, which run_datagen_pipeline.py
    # passes unconditionally. --num_distractors is accepted for the same reason
    # but has no meaning here: --random gives every block a goal predicate, so a
    # block cannot be goal-irrelevant. Use --num-blocks to set the scene size.
    parser.add_argument(
        "--num_distractors",
        type=int,
        default=0,
        help="Not supported by this script; use --num-blocks. Accepted only so "
        "run_datagen_pipeline.py can pass its default of 0.",
    )
    parser.add_argument(
        "--block-size-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Block size range, emitted as a PDDL comment for codegen",
    )
    parser.add_argument(
        "--table-height-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Table height range, emitted as a PDDL comment for codegen",
    )
    parser.add_argument(
        "--friction-range",
        type=float,
        nargs=2,
        default=None,
        metavar=("MIN", "MAX"),
        help="Friction range, emitted as a PDDL comment for codegen",
    )
    args = parser.parse_args()

    if (args.pick_block or args.place_block) and not args.single_pick_place:
        parser.error("--pick-block and --place-block require --single-pick-place")
    if args.num_goal_predicates is not None and not args.random:
        parser.error("--num-goal-predicates requires --random")
    if args.goal_weights is not None and not args.random:
        parser.error("--goal-weights requires --random")
    try:
        goal_weights = parse_goal_weights(args.goal_weights or "")
    except ValueError as exc:
        parser.error(f"--goal-weights: {exc}")
    if args.letter_only and args.color_only:
        parser.error("--letter-only and --color-only are mutually exclusive")

    if args.letter_only:
        COLOR_BLOCKS = []
    elif args.color_only:
        LETTER_BLOCKS = []
    if args.letter_blocks is not None:
        LETTER_BLOCKS = args.letter_blocks
    if args.color_blocks is not None:
        COLOR_BLOCKS = args.color_blocks
    ALL_BLOCKS = LETTER_BLOCKS + COLOR_BLOCKS

    if args.num_distractors:
        parser.error(
            f"--num_distractors={args.num_distractors} is not supported by this script: "
            f"--random gives every block a goal predicate, so no block is goal-irrelevant. "
            f"Use --num-blocks to set the scene size."
        )

    # --num-blocks samples the scene per problem, so the list flags stop being the
    # scene and instead narrow what may be sampled. Left at their defaults, the
    # pools are all 26 letters and every registered colour cube. --require-blocks
    # pins part of the scene; --num-blocks fills whatever is left over.
    scene_pool: list[str] = []
    letter_pool: set[str] = set()
    required_blocks: list[str] = []
    sample_count = 0
    sampled_scene = args.num_blocks is not None or args.require_blocks is not None
    if sampled_scene:
        # Flags that name individual blocks cannot survive a resampled scene: the
        # block they point at is usually absent, which would emit PDDL referencing
        # undeclared objects. --letter-blocks / --color-blocks are fine, since they
        # narrow the pool rather than name scene members.
        conflicting = [
            flag
            for flag, value in (
                ("--goal-spelling", args.goal_spelling),
                ("--init-on-table", args.init_on_table),
                ("--init-clear", args.init_clear),
                ("--init-ori", args.init_ori),
                ("--goal-ori", args.goal_ori),
                ("--pick-block", args.pick_block),
                ("--place-block", args.place_block),
            )
            if value
        ]
        if conflicting:
            parser.error(
                f"--num-blocks samples a different scene per problem, so it cannot be "
                f"combined with {', '.join(conflicting)}, which name specific blocks"
            )
        try:
            required_blocks = expand_block_specs(args.require_blocks or [])
        except ValueError as exc:
            parser.error(f"--require-blocks: {exc}")
        # --goal-condition may co-exist with a sampled scene as long as every
        # block it names is pinned by --require-blocks. That way the goal never
        # references a distractor that this problem happened not to sample, while
        # the remaining scene slots stay random (e.g. R-on-red with distractors).
        if args.goal_condition:
            gc_blocks = set(re.findall(r"block_\w+", " ".join(args.goal_condition)))
            missing = gc_blocks - set(required_blocks)
            if missing:
                parser.error(
                    "--goal-condition with --num-blocks/--require-blocks may only "
                    f"reference required blocks; add {sorted(missing)} to "
                    "--require-blocks so every sampled scene contains them."
                )
        letters = (
            args.letter_blocks
            if args.letter_blocks is not None
            else ([] if args.color_only else LETTER_BLOCK_POOL)
        )
        colors = (
            args.color_blocks
            if args.color_blocks is not None
            else ([] if args.letter_only else COLOR_BLOCK_POOL)
        )
        # Required blocks are already in the scene, so the fill must not draw them
        # again; a second copy of one name is not a legal PDDL object list.
        letter_pool = set(letters) | {
            b for b in required_blocks if _GLYPH_BLOCK_PATTERN.match(b) or b in LEGACY_BLOCKS
        }
        scene_pool = [
            b for b in dict.fromkeys(list(letters) + list(colors)) if b not in required_blocks
        ]
        total = args.num_blocks if args.num_blocks is not None else len(required_blocks)
        sample_count = total - len(required_blocks)
        if total < 1:
            parser.error(f"--num-blocks must be at least 1, got {total}")
        if sample_count < 0:
            parser.error(
                f"--require-blocks asks for {len(required_blocks)} blocks, more than "
                f"--num-blocks {total}"
            )
        if sample_count > len(scene_pool):
            parser.error(
                f"--num-blocks {total} needs {sample_count} more blocks than "
                f"--require-blocks pins, but only {len(scene_pool)} are available "
                f"({len(letters)} letter, {len(colors)} colour, minus those required)"
            )

    if not sampled_scene and not ALL_BLOCKS:
        parser.error("At least one block must be specified (both block lists are empty)")

    init_on_table = set(args.init_on_table) if args.init_on_table else None
    init_clear = set(args.init_clear) if args.init_clear else None
    init_orientations = _parse_key_value_pairs(args.init_ori) if args.init_ori else None
    goal_orientations = _parse_key_value_pairs(args.goal_ori) if args.goal_ori else None

    if init_on_table:
        _validate_blocks(init_on_table, "--init-on-table")
    if init_clear:
        _validate_blocks(init_clear, "--init-clear")
    if init_orientations:
        _validate_blocks(init_orientations.keys(), "--init-ori")
    if goal_orientations:
        _validate_blocks(goal_orientations.keys(), "--goal-ori")
    if args.pick_block:
        _validate_blocks([args.pick_block], "--pick-block")
    if args.place_block:
        _validate_blocks([args.place_block], "--place-block")

    goal_conditions = args.goal_condition

    if args.seed is not None:
        random.seed(args.seed)

    from tqdm import tqdm

    seen = set()
    verbose = args.count == 1

    if goal_conditions:
        full_rounds = args.count // len(goal_conditions)
        remainder = args.count % len(goal_conditions)
        gc_schedule = goal_conditions * full_rounds + random.sample(goal_conditions, remainder)
        random.shuffle(gc_schedule)
    else:
        gc_schedule = [None] * args.count

    for i in tqdm(range(args.count), desc="Generating problems", disable=args.count == 1):
        gc = gc_schedule[i]
        if sampled_scene:
            # Resample every problem so a batch spans many letter/colour mixes
            # rather than repeating one cast. Seeded above, so this stays
            # reproducible for a given --seed.
            scene = required_blocks + random.sample(scene_pool, sample_count)
            LETTER_BLOCKS = [b for b in scene if b in letter_pool]
            COLOR_BLOCKS = [b for b in scene if b not in letter_pool]
            ALL_BLOCKS = LETTER_BLOCKS + COLOR_BLOCKS
        generate_problem(
            out_dir=args.out_dir,
            seed=None,
            seen=seen,
            verbose=verbose,
            init_on_table=init_on_table,
            init_clear=init_clear,
            init_orientations=init_orientations,
            goal_spelling=args.goal_spelling,
            goal_condition=gc,
            goal_orientations=goal_orientations,
            single_pick_place=args.single_pick_place,
            pick_block=args.pick_block,
            place_block=args.place_block,
            hard=args.hard,
            fully_random=args.random,
            num_goal_predicates=args.num_goal_predicates,
            goal_weights=goal_weights,
            rotate=args.rotate,
            flip=args.flip,
            rotate_no_place=args.rotate_no_place,
            easy=args.easy,
            two_skill=args.two_skill,
            block_size_range=tuple(args.block_size_range) if args.block_size_range else None,
            table_height_range=(
                tuple(args.table_height_range) if args.table_height_range else None
            ),
            friction_range=tuple(args.friction_range) if args.friction_range else None,
        )
    print(f"{args.count} requested, {len(seen)} unique problems generated")
