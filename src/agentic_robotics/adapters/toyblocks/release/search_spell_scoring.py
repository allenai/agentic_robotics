"""Success for a Search & Spell task instance, read from the final scene's block poses.

The developer's specification: an episode succeeds if the final state holds a
stack of 3 blocks whose front faces spell the letters of the target word, uppercase or
lowercase, from top to bottom, with those faces oriented so the letters are upright from the
front camera; a block is accepted within 45 degrees of yaw of the nominal orientation. Read here
as:

- *stack*: a block is on another when it rests on it, whatever the offset: one block height
  above it (two cubes resting flat are separated by the mean of their edges, so the height step
  is the edge for the blocks in play) with overlapping footprints, which for cubes means a
  horizontal offset smaller than that height step. The simulator has settled every scored state,
  so a tower that stands is stable; no center tolerance is applied. A stack of 3 is a tower of
  exactly three blocks, its bottom on no other block and nothing on its top; the report also
  says whether three consecutive blocks of a taller tower would have spelled the word
  (`success_any_three`), should the developer read the rule that way instead.
- *front face spells the letter*: the block has a face carrying the letter in either case (its
  family's `letter_faces`, from the block face maps), and the block is within `TILT_TOL_DEG` of
  tilt and `YAW_TOL_DEG` of yaw of the nominal orientation that turns that face to the camera
  with the glyph's top up (`block_orientations.nominal_orientation`), the yaw measured from
  world x or from the line to the camera (`block_orientations.orientation_ok`). Nothing else is
  required: distractors, cubes and the gripper may be anywhere.

The report beside the verdict (`letters_in_place`, `letters_readable`, `tower_heights`, the
readable letters per block) is what lets a near miss be read from an episode summary without
the image. `feasible` and `absent_letter` carry the family's feasibility: on an infeasible
instance no stack can spell the word, so `success` is always False there and the outcome of
interest is the agent's own claim, which the runner records.
``agentic-robotics-spell-check` (drivers/spelling_check.py)` is the known-answer check.
"""

from __future__ import annotations

import math
from functools import cache
from typing import Any

from agentic_robotics.adapters.toyblocks.block_orientations import (
    FRONT_AXIS,
    ORI_QUATERNIONS,
    Quaternion,
    camera_azimuth_deg,
    nominal_orientation,
    orientation_ok,
    world_axis_of,
)
from agentic_robotics.adapters.toyblocks.release.search_spell_families import (
    Family,
    family_of_instance,
)

YAW_TOL_DEG = 45.0
"""The specification's yaw tolerance."""

ON_DZ_MIN_M = 0.025
ON_DZ_MAX_M = 0.09
"""Height steps that can be one block resting on another: the smallest spawned edge is 3.5 cm,
less settling penetration, up to a 5 cm block on a 5 cm block with room for tilt."""


def _poses(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {obj["name"]: obj["pose"] for obj in state.get("objects", [])}


def _quat(pose: dict[str, Any]) -> Quaternion:
    q = pose["orientation"]
    return (q["w"], q["x"], q["y"], q["z"])


def _xy(pose: dict[str, Any]) -> tuple[float, float]:
    return (pose["position"]["x"], pose["position"]["y"])


def _is_on(top: dict[str, Any], bottom: dict[str, Any]) -> bool:
    """`top` rests on `bottom`: one block height above it with overlapping footprints. The
    height step of two cubes resting flat is the mean of their edges, so the footprints overlap
    exactly when each horizontal offset is smaller than the height step."""
    dx = abs(top["position"]["x"] - bottom["position"]["x"])
    dy = abs(top["position"]["y"] - bottom["position"]["y"])
    dz = top["position"]["z"] - bottom["position"]["z"]
    return ON_DZ_MIN_M <= dz <= ON_DZ_MAX_M and dx < dz and dy < dz


BRIDGE_TOL_M = 0.005
"""Blocks beneath a block within this much of its nearest support's height step all support it."""


def supports(poses: dict[str, dict[str, Any]], names: list[str], top: str) -> list[str]:
    """The blocks `top` rests on: of the blocks it overlaps from above, the nearest beneath it,
    and any other at the same height step, which it bridges. A block two levels down also
    overlaps from above, which is why only the nearest level counts."""
    below = [
        (poses[top]["position"]["z"] - poses[b]["position"]["z"], b)
        for b in names
        if b != top and _is_on(poses[top], poses[b])
    ]
    if not below:
        return []
    nearest = min(dz for dz, _ in below)
    return [b for dz, b in below if dz - nearest <= BRIDGE_TOL_M]


def towers(poses: dict[str, dict[str, Any]], names: list[str]) -> list[list[str]]:
    """Every maximal chain of blocks resting on one another among `names`, each listed top to
    bottom. A block bridging two blocks rests on both, so it heads a chain through each; the
    chains then share blocks, and there can be more chains than blocks on the table."""
    above: dict[str, list[str]] = {name: [] for name in names}
    supported: set[str] = set()
    for top in names:
        for bottom in supports(poses, names, top):
            above[bottom].append(top)
            supported.add(top)
    result: list[list[str]] = []

    def climb(chain: list[str]) -> None:
        tops = above[chain[-1]]
        if not tops:
            result.append(list(reversed(chain)))
        for top in tops:
            climb([*chain, top])

    for name in names:
        if name not in supported:
            climb([name])
    return result


SYMMETRIC_180: frozenset[str] = frozenset("HIOSXZosxz")
"""Glyphs that read the same turned a half turn in the plane of the face, so a block showing one
of them upside down spells it. The set is read off the `eval_blocks_36_0` decal sheet: its N is not
symmetric, no glyph reads the same turned a quarter (the O and X are taller than wide), and pairs
that are half turns of each other (p/d, b/q, u/n, M/W) are not self-symmetric, so they stay
strict."""


def _opposite(axis: str) -> str:
    return ("-" if axis[0] == "+" else "+") + axis[1]


def glyph_ups(glyph: str, up: str) -> tuple[str, ...]:
    """Every body axis that may point upward for `glyph` to read as itself: its own up, plus the
    opposite one for the half-turn-symmetric glyphs."""
    return (up, _opposite(up)) if glyph in SYMMETRIC_180 else (up,)


def readable_letters(
    family: Family, name: str, pose: dict[str, Any], yaw_tol_deg: float = YAW_TOL_DEG
) -> list[str]:
    """The letters on `name` that read from the front camera in `pose`: the face toward the
    camera with the glyph's top up, or upside down for a half-turn-symmetric glyph."""
    out = []
    for face in family.letter_faces(name):
        accepted = tuple(
            nominal_orientation(face.face, up) for up in glyph_ups(face.glyph, face.up)
        )
        if orientation_ok(_quat(pose), accepted, camera_azimuth_deg(_xy(pose)), yaw_tol_deg):
            out.append(face.glyph)
    return out


def search_spell_report(state: dict[str, Any], family: Family) -> dict[str, Any]:
    """Score a final `SimState` dict against a Search & Spell family; see the module docstring.

    :return: `success`, plus what the scene held: `towers` (top to bottom), `tower_heights`,
        `readable` (letters readable per block), and for the best tower `letters_in_place` and
        `letters_readable` by position (first letter first)
    """
    poses = _poses(state)
    names = [name for name in family.blocks if name in poses]
    word = family.word
    readable = {name: readable_letters(family, name, poses[name]) for name in names}
    carries = {name: {face.glyph.upper() for face in family.letter_faces(name)} for name in names}

    def positions(tower: list[str]) -> tuple[list[bool], list[bool]]:
        in_place = [letter in carries[name] for letter, name in zip(word, tower, strict=True)]
        read = [
            letter in {g.upper() for g in readable[name]}
            for letter, name in zip(word, tower, strict=True)
        ]
        return in_place, read

    all_towers = towers(poses, names)
    best: tuple[int, int, list[bool], list[bool], list[str]] | None = None
    success = False
    success_any_three = False
    for tower in all_towers:
        windows = [tower[i : i + 3] for i in range(len(tower) - 2)] if len(tower) >= 3 else []
        for window in windows:
            in_place, read = positions(window)
            if all(read):
                success_any_three = True
                if len(tower) == 3:
                    success = True
            key = (sum(read), sum(in_place), in_place, read, window)
            if best is None or key[:2] > best[:2]:
                best = key
    return {
        "success": success,
        "success_any_three": success_any_three,
        "feasible": family.feasible,
        "absent_letter": family.absent_letter,
        "word": word,
        "towers": all_towers,
        "tower_heights": [len(t) for t in all_towers],
        "readable": {name: letters for name, letters in readable.items() if letters},
        "best_tower": best[4] if best else [],
        "letters_in_place": best[2] if best else [],
        "letters_readable": best[3] if best else [],
    }


@cache
def family_for_slug(task_slug: str) -> Family | None:
    """The family of a task instance slug, read once per process; None for any other task."""
    return family_of_instance(task_slug)


def search_spell_report_for(state: dict[str, Any], task_slug: str | None) -> dict[str, Any] | None:
    """`search_spell_report` for a task slug, or None when it is not a Search & Spell instance."""
    family = family_for_slug(task_slug) if task_slug else None
    return None if family is None else search_spell_report(state, family)


def nearest_orientation(quat_wxyz: Quaternion) -> tuple[str, float]:
    """The abstract orientation (one of the 24) closest to a block's quaternion, and the angle to it
    in degrees."""
    best: tuple[str, float] | None = None
    for name, target in ORI_QUATERNIONS.items():
        dot = abs(sum(a * b for a, b in zip(quat_wxyz, target, strict=True)))
        angle = math.degrees(2.0 * math.acos(min(1.0, dot)))
        if best is None or angle < best[1]:
            best = (name, angle)
    assert best is not None
    return best


def exposed_target_letters(state: dict[str, Any], family: Family) -> list[str]:
    """The target-letter faces visible in `state` under the specification's rule, as
    `<glyph>@<block>`: a face is visible on the front of a block (its world axis `FRONT_AXIS` under
    the block's nearest abstract orientation) or on top of a block with nothing on it. The rule
    is the one the generator hides the letters under, applied to a continuous state through the
    nearest of the 24 orientations; it says nothing about whether the camera could read the
    glyph."""
    poses = _poses(state)
    names = [name for name in family.blocks if name in poses]
    covered = {bottom for tower in towers(poses, names) for bottom in tower[1:]}
    out = []
    for name in names:
        faces = family.target_faces(name)
        if not faces:
            continue
        ori, _ = nearest_orientation(_quat(poses[name]))
        for face in faces:
            axis = world_axis_of(face.face, ori)
            if axis == FRONT_AXIS or (axis == "+z" and name not in covered):
                out.append(f"{face.glyph}@{name}")
    return out


def declared_outcome(final_message: str | None, declarations: dict[str, str]) -> str | None:
    """What the agent's final message declares: the value of the one substring of `declarations`
    it contains, "both" if it contains more than one, None if none."""
    if not final_message:
        return None
    found = [outcome for token, outcome in declarations.items() if token in final_message]
    if not found:
        return None
    return found[0] if len(found) == 1 else "both"
