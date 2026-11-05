"""Success for the spelling tasks, read from the final scene's block poses.

`spelling_success` is what `score_scene` and `evaluate_suite rescore` call: for a Search & Spell
instance (by task slug) it is the specification's rule,
`agentic_robotics.adapters.toyblocks.release.search_spell_scoring`; otherwise it is the LEAP
AI2 rule below. The orientation test and camera geometry are
`agentic_robotics.adapters.toyblocks.block_orientations`, re-exported here under the names
the probe scripts import.

**AI2**: a tower that spells AI2, Ai2, aI2 or ai2 is solved. The autogen classifier accepts
one orientation per block (`o2`, the uppercase face toward the camera), so it scores a tower
ending with the lowercase face forward as a failure; this is the corrected verdict. Each
alphabet block carries its lowercase glyph on the face opposite the uppercase one, which is
orientation `o4` (`o2` yawed 180 degrees, as `decode_block_faces.py` confirms for every
letter). The 2 block has one accepted face. The on-relations are taken from the classifier's
own conjuncts.
"""

from __future__ import annotations

from typing import Any

from agentic_robotics.adapters.toyblocks.block_orientations import (  # noqa: F401
    FRONT_CAMERA_XY_M,
    ORI_QUATERNIONS,
    TILT_TOL_DEG,
    YAW_TOL_DEG,
    camera_azimuth_deg,
    orientation_errors,
    orientation_ok,
)
from agentic_robotics.adapters.toyblocks.release.search_spell_scoring import (  # noqa: F401
    search_spell_report,
    search_spell_report_for,
)

ACCEPTED_ORIENTATIONS: dict[str, tuple[str, ...]] = {
    "block_a": ("o2", "o4"),
    "block_i": ("o2", "o4"),
    "block_2": ("o2",),
}
"""Accepted `ORI_QUATERNIONS` names per block of the AI2 goal."""


def spelling_success(
    state: dict[str, Any], conjuncts: dict[str, float], task_slug: str | None = None
) -> bool | None:
    """The corrected verdict for a spelling task, from a `SimState` dict.

    :param state: The final `SimState`, as a dict
    :param conjuncts: The package classifier's conjuncts, for the AI2 rule's on-relations
    :param task_slug: The task instance's slug; a Search & Spell instance is scored by its rule
    :return: None when the state belongs to neither spelling task
    """
    report = search_spell_report_for(state, task_slug)
    if report is not None:
        return report["success"]

    poses = {obj["name"]: obj["pose"] for obj in state.get("objects", [])}
    if not all(name in poses for name in ACCEPTED_ORIENTATIONS):
        return None
    on_ok = conjuncts.get("g0_conj_0") == 1 and conjuncts.get("g0_conj_1") == 1
    if not on_ok:
        return False
    return all(
        orientation_ok(
            (q["w"], q["x"], q["y"], q["z"]),
            accepted,
            camera_azimuth_deg((poses[name]["position"]["x"], poses[name]["position"]["y"])),
        )
        for name, accepted in ACCEPTED_ORIENTATIONS.items()
        for q in [poses[name]["orientation"]]
    )
