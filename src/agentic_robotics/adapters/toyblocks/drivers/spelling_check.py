"""Known-answer check of the Search & Spell scorer
(`agentic_robotics.adapters.toyblocks.release.search_spell_scoring`).

The verification the developer's specification calls for (CLAUDE.md, "Verification, not
speculative tests"): `SimState` dicts built by hand, each with a verdict the specification
fixes, scored and compared. Stdlib only; runs under either interpreter::

    agentic-robotics-spell-check

Two synthetic families stand in for the two collections: a patterned one whose blocks carry the
uppercase letter on body +y and the lowercase on -y, both reading upright with +z up (the
decoded `train_blocks_large` layout), and an unpatterned one with the face geometry of three
real `eval_blocks_36_0` blocks (A on +y with -x up, U on -y with +z up, k on +z with -x up).
Blocks are 4 cm, the table top is at z = 0.
"""

from __future__ import annotations

import sys
from typing import Any

from agentic_robotics.adapters.toyblocks import runner_bootstrap as boot

boot.prepare_import_path()

from agentic_robotics.adapters.toyblocks.block_orientations import (
    ORI_QUATERNIONS,
    nominal_orientation,
    yawed,
)
from agentic_robotics.adapters.toyblocks.leap.spelling import search_spell_report, spelling_success
from agentic_robotics.adapters.toyblocks.release.search_spell_families import Family

SIZE = 0.04


def _family(word: str, blocks: dict[str, dict[str, dict[str, str]]]) -> Family:
    return Family(
        slug=f"spell_{word.lower()}_test_0_feasible",
        word=word,
        collection="test",
        library="test",
        inventory=0,
        feasibility="feasible",
        absent_letter=None,
        targets=tuple(list(blocks)[:3]),
        blocks={
            name: {"kind": "letter", "type": "block_letter", "letter_faces": faces}
            for name, faces in blocks.items()
        },
        on={},
        on_table=tuple(blocks),
        ori={},
        instances=(f"spell_{word.lower()}_test_0_feasible_0",),
    )


def patterned(letter: str) -> dict[str, dict[str, str]]:
    return {"+y": {"glyph": letter, "up": "+z"}, "-y": {"glyph": letter.lower(), "up": "+z"}}


PATTERNED = _family("AUK", {f"block_{c}": patterned(c.upper()) for c in "aukxyz"})
UNPATTERNED = _family(
    "AUK",
    {
        "block_00021": {"+y": {"glyph": "A", "up": "-x"}, "+z": {"glyph": "X", "up": "+x"}},
        "block_00034": {"-y": {"glyph": "U", "up": "+z"}},
        "block_00020": {"+z": {"glyph": "k", "up": "-x"}, "-y": {"glyph": "n", "up": "+x"}},
        "block_00007": {},
    },
)


# The face geometry of three real `eval_blocks_36_0` blocks: O on +z with +x up, W on +z with
# -x up, L on -z with +x up.
OWL = _family(
    "OWL",
    {
        "block_00001": {"+z": {"glyph": "O", "up": "+x"}, "-x": {"glyph": "Y", "up": "+z"}},
        "block_00032": {"+z": {"glyph": "W", "up": "-x"}, "+x": {"glyph": "K", "up": "+z"}},
        "block_00008": {"-z": {"glyph": "L", "up": "+x"}},
    },
)


def block(name: str, xyz: tuple[float, float, float], quat: tuple) -> dict[str, Any]:
    return {
        "name": name,
        "pose": {
            "position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
            "orientation": {"w": quat[0], "x": quat[1], "y": quat[2], "z": quat[3]},
        },
    }


def tower(
    names_top_down: list[str],
    oris: list[str],
    xy: tuple[float, float] = (0.45, 0.0),
    yaw: float = 0.0,
) -> list[dict[str, Any]]:
    """Blocks stacked at `xy`, bottom at table height, each at its orientation yawed by `yaw`."""
    out = []
    for level, (name, ori) in enumerate(zip(reversed(names_top_down), reversed(oris), strict=True)):
        z = SIZE / 2 + level * SIZE
        out.append(block(name, (xy[0], xy[1], z), yawed(ORI_QUATERNIONS[ori], yaw)))
    return out


def state(*blocks: list[dict[str, Any]]) -> dict[str, Any]:
    return {"objects": [b for group in blocks for b in group]}


def case(label: str, report: dict[str, Any], expect: bool, **fields: Any) -> bool:
    ok = report["success"] == expect and all(report[k] == v for k, v in fields.items())
    mark = "ok  " if ok else "FAIL"
    print(
        f"{mark} {label}: success={report['success']} (expected {expect}) "
        f"in_place={report['letters_in_place']} readable={report['letters_readable']} "
        f"heights={report['tower_heights']}" + (f" {fields}" if not ok else "")
    )
    return ok


def main() -> int:
    results = []
    far = [block("block_x", (0.3, 0.25, SIZE / 2), ORI_QUATERNIONS["o1"])]
    up = ["o2", "o2", "o2"]  # uppercase forward
    low = ["o4", "o4", "o4"]  # lowercase forward

    s = state(tower(["block_a", "block_u", "block_k"], up), far)
    results.append(
        case(
            "AUK uppercase, top to bottom",
            search_spell_report(s, PATTERNED),
            True,
            letters_in_place=[True, True, True],
            letters_readable=[True, True, True],
        )
    )
    s = state(tower(["block_a", "block_u", "block_k"], low), far)
    results.append(case("auk lowercase faces", search_spell_report(s, PATTERNED), True))

    # Symmetric glyphs: an O reads upside down but not turned a quarter; a W does not read
    # upside down.
    owl_up = [
        nominal_orientation("+z", "+x"),
        nominal_orientation("+z", "-x"),
        nominal_orientation("-z", "+x"),
    ]
    owl_blocks = ["block_00001", "block_00032", "block_00008"]
    s = state(tower(owl_blocks, owl_up))
    results.append(case("OWL nominal", search_spell_report(s, OWL), True))
    s = state(tower(owl_blocks, [nominal_orientation("+z", "-x"), *owl_up[1:]]))
    results.append(
        case(
            "OWL with the O upside down",
            search_spell_report(s, OWL),
            True,
            letters_readable=[True, True, True],
        )
    )
    s = state(tower(owl_blocks, [nominal_orientation("+z", "+y"), *owl_up[1:]]))
    results.append(
        case(
            "OWL with the O turned a quarter",
            search_spell_report(s, OWL),
            False,
            letters_readable=[False, True, True],
        )
    )
    s = state(tower(owl_blocks, [owl_up[0], nominal_orientation("+z", "+x"), owl_up[2]]))
    results.append(
        case(
            "OWL with the W upside down",
            search_spell_report(s, OWL),
            False,
            letters_readable=[True, False, True],
        )
    )
    s = state(tower(["block_a", "block_u", "block_k"], ["o2", "o4", "o2"]), far)
    results.append(case("AuK mixed case", search_spell_report(s, PATTERNED), True))
    s = state(tower(["block_k", "block_u", "block_a"], up), far)
    results.append(
        case(
            "KUA, bottom-to-top reading",
            search_spell_report(s, PATTERNED),
            False,
            letters_in_place=[False, True, False],
        )
    )
    s = state(tower(["block_a", "block_k", "block_u"], up), far)
    results.append(case("AKU, two letters swapped", search_spell_report(s, PATTERNED), False))
    s = state(tower(["block_a", "block_u", "block_k"], ["o1", "o2", "o2"]), far)
    results.append(
        case(
            "A edge-on",
            search_spell_report(s, PATTERNED),
            False,
            letters_in_place=[True, True, True],
            letters_readable=[False, True, True],
        )
    )
    s = state(tower(["block_a", "block_u", "block_k"], up, yaw=40.0), far)
    results.append(case("tower yawed 40 deg", search_spell_report(s, PATTERNED), True))
    s = state(tower(["block_a", "block_u", "block_k"], up, yaw=50.0), far)
    results.append(case("tower yawed 50 deg", search_spell_report(s, PATTERNED), False))
    s = state(tower(["block_a", "block_u", "block_k"], up, xy=(0.45, 0.35), yaw=-30.0), far)
    results.append(
        case("tower at y=0.35 turned toward the camera", search_spell_report(s, PATTERNED), True)
    )
    s = state(tower(["block_a", "block_x", "block_u", "block_k"], [*up, "o2"]), far[:0])
    results.append(case("distractor inside the tower", search_spell_report(s, PATTERNED), False))
    s = state(tower(["block_a", "block_u", "block_k", "block_x"], [*up, "o2"]))
    results.append(
        case(
            "AUK on a fourth block",
            search_spell_report(s, PATTERNED),
            False,
            success_any_three=True,
            tower_heights=[4],
        )
    )
    s = state(tower(["block_x", "block_a", "block_u", "block_k"], ["o2", *up]))
    results.append(
        case(
            "a fourth block on AUK",
            search_spell_report(s, PATTERNED),
            False,
            success_any_three=True,
        )
    )
    apart = tower(["block_a", "block_u", "block_k"], up)
    apart[0]["pose"]["position"]["x"] += 0.03  # the bottom block 3 cm off: 1 cm of overlap
    results.append(
        case(
            "bottom block 3 cm off the stack, footprints still overlapping",
            search_spell_report(state(apart, far), PATTERNED),
            True,
            tower_heights=[3, 1],
        )
    )
    apart = tower(["block_a", "block_u", "block_k"], up)
    apart[0]["pose"]["position"]["x"] += 0.045  # the bottom block clear of the stack
    results.append(
        case(
            "bottom block 4.5 cm off the stack, no overlap",
            search_spell_report(state(apart, far), PATTERNED),
            False,
            tower_heights=[2, 1, 1],
        )
    )
    s = state(tower(["block_a", "block_u"], up[:2]), far)
    results.append(
        case(
            "only two letters stacked",
            search_spell_report(s, PATTERNED),
            False,
            letters_in_place=[],
            tower_heights=[2, 1],
        )
    )

    oris = [
        nominal_orientation("+y", "-x"),
        nominal_orientation("-y", "+z"),
        nominal_orientation("+z", "-x"),
    ]
    s = state(tower(["block_00021", "block_00034", "block_00020"], oris))
    results.append(
        case(
            "unpatterned AUK at the nominal orientations", search_spell_report(s, UNPATTERNED), True
        )
    )
    s = state(tower(["block_00021", "block_00034", "block_00020"], ["o2", oris[1], oris[2]]))
    results.append(
        case(
            "unpatterned A face forward but rotated 90 deg",
            search_spell_report(s, UNPATTERNED),
            False,
            letters_readable=[False, True, True],
        )
    )
    s = state(tower(["block_00021", "block_00034", "block_00020"], ["o14", oris[1], oris[2]]))
    results.append(
        case("unpatterned A at o14, the decoded nominal", search_spell_report(s, UNPATTERNED), True)
    )

    # A state outside Search & Spell falls through the dispatcher to the AI2 rule.
    ai2 = state(tower(["block_a", "block_i", "block_2"], ["o4", "o2", "o2"]))
    verdict = spelling_success(ai2, {"g0_conj_0": 1.0, "g0_conj_1": 1.0})
    results.append(
        case(
            "AI2 rule: aI2",
            {
                "success": verdict,
                "letters_in_place": None,
                "letters_readable": None,
                "tower_heights": None,
            },
            True,
        )
    )
    results.append(spelling_success(state(far), {}) is None)

    failed = results.count(False)
    print(f"{len(results) - failed} of {len(results)} cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
